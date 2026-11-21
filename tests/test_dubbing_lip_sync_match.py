"""Phase 2 tests: source scan cache, LR-ASD parsing, speaker-to-face matching, and window cache."""

import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

from h3_app.dubbing.alignment import FITTED_AUDIO_RATE as RATE
from h3_app.dubbing.lip_sync import REPORT_FILE, render_dub_lip_sync
from h3_app.dubbing.lip_sync_match import (
    FaceTrack,
    Shot,
    SourceScan,
    TrackObservation,
    assign_speakers_to_shots,
    get_or_scan_source,
    load_source_scan,
    parse_lr_asd_tracks,
    save_source_scan,
    uncertain_assignments,
)
from h3_app.dubbing.results import player_folder
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
            services = SimpleNamespace(
                models=SimpleNamespace(load_model_config=lambda: model_cfg),
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
            # Left box center = (0.2*320, 0.4*240) = (64, 96); Right box center = (0.8*320, 0.4*240) = (256, 96).
            self.assertEqual(submitted_targets, [(64, 96), (256, 96)])
            report1 = json.loads((folder / REPORT_FILE).read_text(encoding="utf-8"))
            self.assertEqual((report1["total_windows"], report1["cached_windows"], report1["rendered_windows"]), (2, 0, 2))

            # Run 2: identical inputs -> 100% cache hit, zero ComfyUI submissions!
            submit_mock.reset_mock()
            list(render_dub_lip_sync(dub=dub, services=services, crop_size=320, scanner=fake_scanner))
            self.assertEqual(submit_mock.call_count, 0)
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
