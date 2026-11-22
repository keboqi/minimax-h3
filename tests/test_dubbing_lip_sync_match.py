"""Phase 2 tests: source scan cache, LR-ASD parsing, speaker-to-face matching, and window cache."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from h3_app.dubbing.alignment import FITTED_AUDIO_RATE as RATE
from h3_app.dubbing.lip_sync import REPORT_FILE, render_dub_lip_sync
from h3_app.dubbing.lip_sync_match import (
    FaceTrack,
    Shot,
    SourceScan,
    TrackObservation,
    _score_motion_series,
    assign_speakers_to_shots,
    estimate_decision_snippet_tokens,
    get_or_scan_source,
    load_source_scan,
    parse_lr_asd_tracks,
    resolve_uncertain_with_gemini_decision,
    save_source_scan,
    uncertain_assignments,
)
from h3_app.dubbing.results import player_folder
from h3_app.llm.decisions import DecisionError
from h3_app.media_tools import probe_video_metadata
from tests.test_face_refine import by_class, models


def _track(track_id: int, shot_index: int, box: tuple[float, float, float, float],
           samples: list[tuple[float, float]]) -> FaceTrack:
    obs = tuple(TrackObservation(time=t, box=box, score=s) for t, s in samples)
    return FaceTrack(track_id=track_id, shot_index=shot_index, start=obs[0].time, end=obs[-1].time, observations=obs)


class LipSyncMatchTests(unittest.TestCase):
    def test_parse_lr_asd_tracks_splits_at_cuts_and_normalizes_boxes(self):
        raw = [{
            "fps": 10.0,
            "frames": [2, 5, 8, 12, 15, 18],
            "bboxes": [[100, 100, 300, 300]] * 6,
            "scores": [1.2, 1.1, 1.0, -0.8, -0.9, -1.0],
        }]
        scan = parse_lr_asd_tracks(
            raw, fingerprint="abc123", duration=2.0, width=1000, height=500, fps=24.0, cuts=[1.0],
        )
        self.assertEqual(len(scan.shots), 2)
        self.assertEqual(len(scan.shots[0].tracks), 1)
        self.assertEqual(len(scan.shots[1].tracks), 1)
        self.assertEqual(scan.shots[0].tracks[0].observations[0].box, (0.1, 0.2, 0.2, 0.4))
        self.assertEqual(len(scan.shots[0].tracks[0].observations), 3)
        self.assertEqual(len(scan.shots[1].tracks[0].observations), 3)

    def test_source_scan_cache_round_trips_and_avoids_rescanning(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "clip.mp4"
            video.write_bytes(b"0" * 1024)
            cache = root / "scans"

            calls = 0

            def fake_scanner(_video, cuts):
                nonlocal calls
                calls += 1
                return SourceScan(
                    fingerprint="", duration=5.0, width=1280, height=720, fps=24.0,
                    cuts=tuple(cuts), shots=(Shot(index=0, start=0.0, end=5.0, tracks=()),),
                    backend="lr-asd",
                )

            # Save directly with real fingerprint via get_or_scan_source.
            from h3_app.dubbing.lip_sync_match import video_fingerprint
            fp = video_fingerprint(video)

            def fp_scanner(v, cuts):
                scan = fake_scanner(v, cuts)
                return SourceScan(
                    fingerprint=fp, duration=scan.duration, width=scan.width, height=scan.height,
                    fps=scan.fps, cuts=scan.cuts, shots=scan.shots, backend=scan.backend,
                )

            first = get_or_scan_source(video, [2.0], cache_dir=cache, scanner=fp_scanner)
            second = get_or_scan_source(video, [2.0], cache_dir=cache, scanner=fp_scanner)
            self.assertEqual(calls, 1)
            self.assertEqual(first.to_dict(), second.to_dict())
            loaded = load_source_scan(cache, fp)
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.backend, "lr-asd")

    def test_shared_cache_survives_pruning_and_rechecks_new_lr_asd(self):
        from h3_app.dubbing.lip_sync_match import video_fingerprint
        from h3_app.dubbing.results import PLAYER_DIR, SCAN_CACHE_DIR, prune_players

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.mp4"
            source.write_bytes(b"original video")
            cache = root / PLAYER_DIR / SCAN_CACHE_DIR
            fp = video_fingerprint(source)
            scan = SourceScan(fp, 2, 640, 480, 24, (), (Shot(0, 0, 2),))
            save_source_scan(cache, scan)
            prune_players(root)
            self.assertEqual(load_source_scan(cache, fp), scan)
            with patch.dict(os.environ, {"H3_LR_ASD_DIR": tmp}), \
                    patch("h3_app.dubbing.lip_sync_match.run_lr_asd_scan", return_value=scan) as run:
                get_or_scan_source(source, (), cache_dir=cache)
                run.assert_called_once()

    def test_language_dubs_scan_the_same_original_picture_once(self):
        from h3_app.dubbing.lip_sync import detect_lip_sync_plan
        from h3_app.dubbing.lip_sync_match import video_fingerprint
        from h3_app.dubbing.results import prune_players

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "original.mov"
            source.write_bytes(b"source picture and soundtrack")
            scan = SourceScan(video_fingerprint(source), 2, 640, 480, 24, (), (Shot(0, 0, 2),), "lr-asd")
            scanner = Mock(return_value=scan)
            with patch("h3_app.dubbing.lip_sync.detect_shot_cuts", return_value=[]), \
                    patch("h3_app.dubbing.lip_sync.probe_video_metadata", return_value=scan):
                for lang in ("en", "fr"):
                    dub = root / f"clip.{lang}.mp4"
                    dub.write_bytes(lang.encode())
                    folder = player_folder(dub)
                    folder.mkdir(parents=True)
                    (folder / "alignment.json").write_text(json.dumps({"lines": [], "spoken": {}}))
                    np.savez(folder / "alignment.npz")
                    prune_players(root)
                    list(detect_lip_sync_plan(dub=dub, source_video=source, scanner=scanner))
            scanner.assert_called_once_with(source, [])

    def test_assign_speakers_handles_two_speaker_lr_asd_reaction_shot_and_ambiguous_case(self):
        # Shot 0 (0..4s): two-person interview. A speaks 0.2..1.8 (left face active), B speaks 2.0..3.6 (right face active).
        left_face = _track(0, 0, (0.15, 0.2, 0.2, 0.4), [
            (0.4, 1.4), (0.8, 1.3), (1.2, 1.5), (1.6, 1.1),
            (2.2, -0.9), (2.6, -0.8), (3.0, -1.0), (3.4, -0.7),
        ])
        right_face = _track(1, 0, (0.65, 0.2, 0.2, 0.4), [
            (0.4, -0.8), (0.8, -0.9), (1.2, -0.7), (1.6, -0.6),
            (2.2, 1.1), (2.6, 1.2), (3.0, 1.0), (3.4, 1.3),
        ])
        # Shot 1 (4..6s): reaction shot — one face visible, silent while A speaks off-screen.
        silent_face = _track(0, 1, (0.4, 0.2, 0.2, 0.4), [
            (4.2, -1.1), (4.6, -1.0), (5.0, -1.2), (5.4, -0.9),
        ])
        # Shot 2 (6..8s): ambiguous — two faces have nearly tied positive scores.
        amb_left = _track(0, 2, (0.2, 0.2, 0.2, 0.4), [(6.2, 0.5), (6.6, 0.55), (7.0, 0.45)])
        amb_right = _track(1, 2, (0.6, 0.2, 0.2, 0.4), [(6.2, 0.42), (6.6, 0.48), (7.0, 0.40)])

        scan = SourceScan(
            fingerprint="fp", duration=8.0, width=1280, height=720, fps=24.0, cuts=(4.0, 6.0),
            shots=(
                Shot(index=0, start=0.0, end=4.0, tracks=(left_face, right_face)),
                Shot(index=1, start=4.0, end=6.0, tracks=(silent_face,)),
                Shot(index=2, start=6.0, end=8.0, tracks=(amb_left, amb_right)),
            ),
            backend="lr-asd",
        )
        lines = [
            {"speaker": "A", "start": 0.2, "end": 1.8},
            {"speaker": "B", "start": 2.0, "end": 3.6},
            {"speaker": "A", "start": 4.1, "end": 5.5},
            {"speaker": "B", "start": 6.1, "end": 7.5},
        ]
        assigned = assign_speakers_to_shots(scan, lines)
        self.assertEqual(assigned[(0, "A")].status, "matched")
        self.assertEqual(assigned[(0, "A")].track_id, 0)
        self.assertEqual(assigned[(0, "B")].status, "matched")
        self.assertEqual(assigned[(0, "B")].track_id, 1)
        # Reaction shot is off-screen, not wrongly patched.
        self.assertEqual(assigned[(1, "A")].status, "off_screen")
        # Ambiguous shot is flagged uncertain for optional review.
        self.assertEqual(assigned[(2, "B")].status, "uncertain")
        self.assertEqual([(u.shot_index, u.speaker) for u in uncertain_assignments(assigned)], [(2, "B")])

        # Manual override resolves shot 2.
        overridden = assign_speakers_to_shots(scan, lines, manual_overrides={(2, "B"): 1, (0, "A"): None})
        self.assertEqual(overridden[(2, "B")].status, "manual")
        self.assertEqual(overridden[(2, "B")].track_id, 1)
        self.assertEqual(overridden[(0, "A")].status, "off_screen")
        self.assertEqual(uncertain_assignments(overridden), [])

    def test_gemini_hints_and_no_asd_multi_speaker_safety(self):
        scan = SourceScan(
            fingerprint="fp", duration=4.0, width=1280, height=720, fps=24.0, cuts=(),
            shots=(Shot(index=0, start=0.0, end=4.0, tracks=()),),
            backend="none",
        )
        # Two speakers with no ASD and no hints -> both uncertain (never blindly patch largest_face).
        no_hints = assign_speakers_to_shots(scan, [
            {"speaker": "A", "start": 0.2, "end": 1.5},
            {"speaker": "B", "start": 2.0, "end": 3.5},
        ])
        self.assertEqual(no_hints[(0, "A")].status, "uncertain")
        self.assertEqual(no_hints[(0, "B")].status, "uncertain")

        # With Gemini hints -> A matched on-screen, B marked off-screen.
        with_hints = assign_speakers_to_shots(scan, [
            {"speaker": "A", "start": 0.2, "end": 1.5,
             "visual_speaker": {"time": 0.8, "visibility": "on_screen", "face_center": [0.3, 0.4]}},
            {"speaker": "B", "start": 2.0, "end": 3.5,
             "visual_speaker": {"time": 2.5, "visibility": "off_screen"}},
        ])
        self.assertEqual(with_hints[(0, "A")].status, "matched")
        self.assertEqual(with_hints[(0, "A")].source, "gemini")
        self.assertEqual(with_hints[(0, "A")].face_center, (0.3, 0.4))
        self.assertEqual(with_hints[(0, "B")].status, "off_screen")

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg/ffprobe required")
    def test_render_dub_lip_sync_caches_windows_and_rerenders_only_overridden_shot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dub = root / "interview.mp4"
            subprocess.run([
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-f", "lavfi", "-i", "color=c=navy:s=320x240:r=24:d=3.0",
                "-f", "lavfi", "-i", "sine=frequency=440:duration=3.0",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
                str(dub),
            ], check=True)
            folder = player_folder(dub)
            folder.mkdir(parents=True)
            samples = lambda s, e: round(e * RATE) - round(s * RATE)
            (folder / "alignment.json").write_text(json.dumps({
                "lines": [
                    {"speaker": "A", "start": 0.2, "end": 1.0},
                    {"speaker": "B", "start": 1.6, "end": 2.4},
                ],
                "spoken": {"0": [0.2, 1.0], "1": [1.6, 2.4]},
            }), encoding="utf-8")
            np.savez_compressed(folder / "alignment.npz", **{
                "0": np.full(samples(0.2, 1.0), 0.25, dtype=np.float32),
                "1": np.full(samples(1.6, 2.4), 0.35, dtype=np.float32),
            })

            from h3_app.dubbing.lip_sync_match import video_fingerprint
            fp = video_fingerprint(dub)
            left_track = _track(0, 0, (0.1, 0.2, 0.2, 0.4), [
                (0.3, 1.2), (0.6, 1.3), (0.9, 1.1), (1.8, -0.9), (2.0, -0.8), (2.2, -0.7),
            ])
            right_track = _track(1, 0, (0.7, 0.2, 0.2, 0.4), [
                (0.3, -0.8), (0.6, -0.9), (0.9, -0.7), (1.8, 1.1), (2.0, 1.2), (2.2, 1.0),
            ])

            def fake_scanner(_video, cuts):
                return SourceScan(
                    fingerprint=fp, duration=3.0, width=320, height=240, fps=24.0,
                    cuts=tuple(cuts), shots=(Shot(index=0, start=0.0, end=3.0, tracks=(left_track, right_track)),),
                    backend="lr-asd",
                )

            out_dir = root / "comfy_out"
            out_dir.mkdir()
            staged_videos: dict[str, Path] = {}
            rendered_map: dict[str, Path] = {}
            submitted_targets: list[tuple[int, int]] = []

            def stage(path, category):
                src = Path(path)
                if src.suffix == ".mp4":
                    copied = out_dir / f"staged_{len(staged_videos)}.mp4"
                    shutil.copy2(src, copied)
                    staged_videos[copied.name] = copied
                    return copied.name
                return src.name

            def submit(graph, client_id):
                track_node = by_class(graph, "H3FaceTrackCrop")
                self.assertEqual(track_node["select"], "closest_to_xy")
                submitted_targets.append((track_node["X"], track_node["Y"]))
                video_name = by_class(graph, "LoadVideo")["file"]
                rendered = out_dir / f"{client_id}_00001_.mp4"
                shutil.copy2(staged_videos[video_name], rendered)
                rendered_map[client_id] = rendered
                return client_id

            model_cfg = models()
            model_cfg.turbo_lora_for = lambda _m, _v: "pika.safetensors"
            submit_mock = Mock(side_effect=submit)
            model_services = Mock(load_model_config=Mock(return_value=model_cfg))
            services = SimpleNamespace(
                models=model_services,
                media=SimpleNamespace(
                    probe_video_metadata=probe_video_metadata,
                    stage_file=stage,
                    resolve_output=lambda history, _: rendered_map[history],
                ),
                execution=SimpleNamespace(
                    submit_prompt=submit_mock,
                    wait_for_history=lambda prompt_id: prompt_id,
                    poll_comfy_progress=lambda *_: [],
                ),
            )

            # Run 1: renders both A (left face) and B (right face).
            list(render_dub_lip_sync(dub=dub, services=services, crop_size=320, scanner=fake_scanner))
            self.assertEqual(submit_mock.call_count, 2)
            model_services.ensure_base_video_vae.assert_called_once_with(model_cfg)
            # Left box center = (0.2*320, 0.4*240) = (64, 96); Right box center = (0.8*320, 0.4*240) = (256, 96).
            self.assertEqual(submitted_targets, [(64, 96), (256, 96)])
            report1 = json.loads((folder / REPORT_FILE).read_text(encoding="utf-8"))
            self.assertEqual((report1["total_windows"], report1["cached_windows"], report1["rendered_windows"]), (2, 0, 2))

            # Run 2: identical inputs -> 100% cache hit, zero ComfyUI submissions!
            submit_mock.reset_mock()
            model_services.ensure_base_video_vae.reset_mock()
            list(render_dub_lip_sync(dub=dub, services=services, crop_size=320, scanner=fake_scanner))
            self.assertEqual(submit_mock.call_count, 0)
            model_services.ensure_base_video_vae.assert_not_called()
            report2 = json.loads((folder / REPORT_FILE).read_text(encoding="utf-8"))
            self.assertEqual((report2["total_windows"], report2["cached_windows"], report2["rendered_windows"]), (2, 2, 0))

            # Run 3: user overrides Speaker B to track 0 -> Speaker A stays cached, ONLY Speaker B re-renders!
            submit_mock.reset_mock()
            submitted_targets.clear()
            list(render_dub_lip_sync(
                dub=dub, services=services, crop_size=320, scanner=fake_scanner,
                manual_overrides={(0, "B"): 0},
            ))
            self.assertEqual(submit_mock.call_count, 1)
            self.assertEqual(submitted_targets, [(64, 96)])
            report3 = json.loads((folder / REPORT_FILE).read_text(encoding="utf-8"))
            self.assertEqual((report3["total_windows"], report3["cached_windows"], report3["rendered_windows"]), (2, 1, 1))

            # Run 4: re-running without passing overrides again still remembers the saved override (100% cache hit).
            submit_mock.reset_mock()
            list(render_dub_lip_sync(dub=dub, services=services, crop_size=320, scanner=fake_scanner))
            self.assertEqual(submit_mock.call_count, 0)

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg/ffprobe required")
    def test_no_face_window_is_skipped_without_failing_other_windows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dub = root / "broll.mp4"
            subprocess.run([
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-f", "lavfi", "-i", "color=c=navy:s=320x240:r=24:d=3.0",
                "-f", "lavfi", "-i", "sine=frequency=440:duration=3.0",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
                str(dub),
            ], check=True)
            folder = player_folder(dub)
            folder.mkdir(parents=True)
            samples = lambda s, e: round(e * RATE) - round(s * RATE)
            (folder / "alignment.json").write_text(json.dumps({
                "lines": [
                    {"speaker": "A", "start": 0.1, "end": 0.7},
                    {"speaker": "A", "start": 2.0, "end": 2.6},
                ],
                "spoken": {"0": [0.1, 0.7], "1": [2.0, 2.6]},
            }), encoding="utf-8")
            np.savez_compressed(folder / "alignment.npz", **{
                "0": np.full(samples(0.1, 0.7), 0.25, dtype=np.float32),
                "1": np.full(samples(2.0, 2.6), 0.25, dtype=np.float32),
            })

            out_dir = root / "comfy_out"
            out_dir.mkdir()
            staged_videos: dict[str, Path] = {}
            rendered_map: dict[str, Path] = {}
            calls = 0

            def stage(path, category):
                src = Path(path)
                if src.suffix == ".mp4":
                    copied = out_dir / f"staged_{len(staged_videos)}.mp4"
                    shutil.copy2(src, copied)
                    staged_videos[copied.name] = copied
                    return copied.name
                return src.name

            def submit(graph, client_id):
                nonlocal calls
                calls += 1
                if calls == 1:
                    raise RuntimeError("No detected face frames available in the active lip-sync interval")
                video_name = by_class(graph, "LoadVideo")["file"]
                rendered = out_dir / f"{client_id}_00001_.mp4"
                shutil.copy2(staged_videos[video_name], rendered)
                rendered_map[client_id] = rendered
                return client_id

            model_cfg = models()
            model_cfg.turbo_lora_for = lambda _m, _v: "pika.safetensors"
            services = SimpleNamespace(
                models=Mock(load_model_config=Mock(return_value=model_cfg)),
                media=SimpleNamespace(
                    probe_video_metadata=probe_video_metadata,
                    stage_file=stage,
                    resolve_output=lambda history, _: rendered_map[history],
                ),
                execution=SimpleNamespace(
                    submit_prompt=submit,
                    wait_for_history=lambda prompt_id: prompt_id,
                    poll_comfy_progress=lambda *_: [],
                ),
            )
            updates = list(render_dub_lip_sync(dub=dub, services=services, crop_size=320, use_cache=False))
            self.assertTrue(updates[-1].is_file())
            report = json.loads((folder / REPORT_FILE).read_text(encoding="utf-8"))
            self.assertEqual((report["total_windows"], report["skipped_windows"], report["rendered_windows"]), (2, 1, 1))

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg required")
    def test_gemini_decision_tie_breaker_resolves_caches_respects_budget_and_falls_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "clip.mp4"
            orig_audio = root / "original.wav"
            subprocess.run([
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-f", "lavfi", "-i", "color=c=navy:s=320x240:r=24:d=4.0",
                "-f", "lavfi", "-i", "sine=frequency=440:duration=4.0",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
                str(video),
            ], check=True)
            subprocess.run([
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-f", "lavfi", "-i", "sine=frequency=660:duration=4.0",
                str(orig_audio),
            ], check=True)
            cache_dir = root / "scans"
            t0 = _track(0, 0, (0.1, 0.2, 0.2, 0.4), [(0.5, 0.4), (1.0, 0.4), (1.5, 0.4)])
            t1 = _track(1, 0, (0.7, 0.2, 0.2, 0.4), [(0.5, 0.35), (1.0, 0.35), (1.5, 0.35)])
            t2 = _track(0, 1, (0.1, 0.2, 0.2, 0.4), [(2.5, 0.4), (3.0, 0.4), (3.5, 0.4)])
            t3 = _track(1, 1, (0.7, 0.2, 0.2, 0.4), [(2.5, 0.35), (3.0, 0.35), (3.5, 0.35)])
            scan = SourceScan(
                fingerprint="fp-dec", duration=4.0, width=320, height=240, fps=24.0, cuts=(2.0,),
                shots=(
                    Shot(index=0, start=0.0, end=2.0, tracks=(t0, t1)),
                    Shot(index=1, start=2.0, end=4.0, tracks=(t2, t3)),
                ),
                backend="lr-asd",
            )
            lines = [
                {"speaker": "A", "start": 0.2, "end": 1.0, "source": "Hello there"},
                {"speaker": "B", "start": 1.1, "end": 1.9, "source": "Narrator voice"},
                {"speaker": "A", "start": 2.2, "end": 3.6, "source": "Hard to tell"},
            ]
            initial = assign_speakers_to_shots(scan, lines)
            self.assertEqual(len(uncertain_assignments(initial)), 3)

            seen_calls = []

            def fake_decide(questions, blocks, *, key, timeout=120):
                seen_calls.append((list(questions.keys()), len(blocks), key))
                return {
                    "s0_A": {"type": "choice", "choice": "face_0",
                             "probabilities": {"face_0": 0.88, "face_1": 0.06, "off_screen": 0.04, "uncertain": 0.02}},
                    "s0_B": {"type": "choice", "choice": "off_screen",
                             "probabilities": {"off_screen": 0.84, "face_0": 0.08, "face_1": 0.05, "uncertain": 0.03}},
                    "s1_A": {"type": "choice", "choice": "face_0",
                             "probabilities": {"face_0": 0.52, "face_1": 0.45, "off_screen": 0.02, "uncertain": 0.01}},
                }

            with patch.dict(os.environ, {"H3_DECISION": "1"}):
                resolved = resolve_uncertain_with_gemini_decision(
                    video=video, scan=scan, lines=lines, assignments=initial,
                    original_audio=orig_audio, api_key="test-key", cache_dir=cache_dir, decide_fn=fake_decide,
                )
                self.assertEqual(len(seen_calls), 1)
                self.assertEqual(set(seen_calls[0][0]), {"s0_A", "s0_B", "s1_A"})
                self.assertEqual(resolved[(0, "A")].status, "matched")
                self.assertEqual(resolved[(0, "A")].track_id, 0)
                self.assertEqual(resolved[(0, "A")].source, "gemini-decision")
                self.assertEqual(resolved[(0, "B")].status, "off_screen")
                self.assertEqual(resolved[(0, "B")].source, "gemini-decision")
                # Low-margin shot stays uncertain for optional review.
                self.assertEqual(resolved[(1, "A")].status, "uncertain")

                # Second call hits cache for s0_A and s0_B -> only queries s1_A.
                seen_calls.clear()
                resolve_uncertain_with_gemini_decision(
                    video=video, scan=scan, lines=lines, assignments=initial,
                    api_key="test-key", cache_dir=cache_dir, decide_fn=fake_decide,
                )
                self.assertEqual(seen_calls[0][0], ["s1_A"])

                # Token budget cap: only 1 snippet fits in a tight budget when max_batches=1.
                seen_calls.clear()
                one_clip_budget = 250 + estimate_decision_snippet_tokens(1.0, 2) + 50
                resolve_uncertain_with_gemini_decision(
                    video=video, scan=scan, lines=lines, assignments=initial,
                    api_key="test-key", cache_dir=None, max_tokens_per_batch=one_clip_budget,
                    max_batches=1, decide_fn=fake_decide,
                )
                self.assertEqual(len(seen_calls), 1)
                self.assertEqual(len(seen_calls[0][0]), 1)

                # DecisionError (e.g. 429 rate limit) falls back cleanly without raising.
                def fail_429(*_args, **_kwargs):
                    raise DecisionError("Rate limited", status=429)

                fallback = resolve_uncertain_with_gemini_decision(
                    video=video, scan=scan, lines=lines, assignments=initial,
                    api_key="test-key", cache_dir=None, decide_fn=fail_429,
                )
                self.assertEqual(fallback, initial)

    def test_score_motion_series_keeps_fallback_diffs_separate_from_mar(self):
        # Steady fallback grayscale diffs (~0.08) represent active speaking and must not be double-differenced to 0.
        fallback_scores = _score_motion_series([None, None, None, None], [None, 0.08, 0.08, 0.08])
        self.assertTrue(all(score > 0.8 for score in fallback_scores))

        # Quiet MAR interleaved with a quiet fallback frame must stay negative (no cross-unit jump spike).
        mixed_scores = _score_motion_series([0.25, 0.255, None, 0.25], [None, None, 0.01, None])
        self.assertTrue(all(score < 0.0 for score in mixed_scores))

    def test_get_or_scan_source_force_rescan_skips_extra_probe(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "clip.mp4"
            video.write_bytes(b"fake-mp4")
            expected = SourceScan("fp", 2.0, 320, 240, 24.0, (), (Shot(0, 0.0, 2.0),), "test")
            with patch("h3_app.dubbing.lip_sync_match.probe_video_metadata") as probe:
                scan = get_or_scan_source(
                    video, (), cache_dir=root / "cache", scanner=lambda _v, _c: expected, force_rescan=True,
                )
                self.assertEqual(scan, expected)
                probe.assert_not_called()

