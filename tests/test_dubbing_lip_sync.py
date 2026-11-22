"""Frozen dialogue and graph wiring for a prepared H3 lip-sync shot."""

import io
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import wave

import numpy as np

from h3_app.dubbing.alignment import FITTED_AUDIO_RATE as RATE
from h3_app.dubbing.lip_sync import (
    LipSyncWindow,
    PLAN_FILE,
    REPORT_FILE,
    _extract_window_slice,
    assemble_lip_sync_video,
    build_lip_sync_plan_payload,
    detect_lip_sync_plan,
    filter_lip_sync_windows,
    fitted_dialogue,
    load_lip_sync_plan,
    plan_lip_sync_windows,
    render_dub_lip_sync,
    render_lip_sync_preview,
    revise_lip_sync_plan,
    validate_face_size_range,
    validate_lip_sync_engine,
    window_cache_key,
)
from h3_app.dubbing.lip_sync_match import FaceTrack, Shot, ShotSpeakerAssignment, SourceScan, TrackObservation
from h3_app.dubbing import lip_sync_worker
from h3_app.dubbing.results import player_folder
from h3_app.errors import H3Error
from h3_app.jobs import JobCancelled
from h3_app.media_tools import probe_video_metadata
from h3_app.workflows.lip_sync import build_auto_lip_sync_graph, build_lip_sync_graph, snap_h3_frames
from tests.test_face_refine import by_class, link_id, models


class LipSyncTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.dub = Path(temporary.name) / "finished.mp4"
        self.dub.touch()
        folder = player_folder(self.dub)
        folder.mkdir(parents=True)
        (folder / "alignment.json").write_text(json.dumps({
            "lines": [{"speaker": "A"}, {"speaker": "B"}, {"speaker": "A"}],
            "spoken": {"0": [0.125, 0.25], "1": [0.0, 0.25], "2": [0.375, 0.5]},
        }), encoding="utf-8")
        samples = lambda start, end: round(end * RATE) - round(start * RATE)
        np.savez_compressed(folder / "alignment.npz", **{
            "0": np.full(samples(0.125, 0.25), 0.25, dtype=np.float32),
            "1": np.full(samples(0.0, 0.25), 0.75, dtype=np.float32),
            "2": np.full(samples(0.375, 0.5), 0.5, dtype=np.float32),
        })
        self.models = models()
        self.models.turbo_lora_for = lambda _mode, _variant: "pika.safetensors"

    def test_dialogue_starts_at_the_fitted_time_and_excludes_other_speakers(self):
        dialogue = fitted_dialogue(self.dub, "A", first_frame=4, frames=8)
        self.assertEqual(len(dialogue), round(RATE / 2) - round(4 * RATE / 24))
        first_interval = round(0.25 * RATE) - round(4 * RATE / 24)
        next_interval = round(0.375 * RATE) - round(4 * RATE / 24)
        np.testing.assert_array_equal(dialogue[:first_interval], 0.25)
        np.testing.assert_array_equal(dialogue[first_interval:next_interval], 0)
        np.testing.assert_array_equal(dialogue[next_interval:], 0.5)
        with self.assertRaisesRegex(H3Error, "no audible fitted dialogue"):
            fitted_dialogue(self.dub, "A", first_frame=24, frames=5)

    def test_graph_keeps_dialogue_locked_and_composites_with_the_original_mask(self):
        graph = build_lip_sync_graph(crop="crop.mp4", mask="mask.mp4", dialogue="dialogue.wav",
                                    width=640, height=640, frames=124, models=self.models,
                                    refinement={}, output_prefix="preview")
        held = by_class(graph, "H3SourceAudioLatent")
        mask = by_class(graph, "H3LipSyncMask")
        self.assertEqual(held["audio"], [link_id(graph, "LoadAudio"), 0])
        self.assertEqual(mask["latent"], [link_id(graph, "H3SourceAudioLatent"), 0])
        sample = by_class(graph, "SamplerCustomAdvanced")
        self.assertEqual(sample["latent_image"], [link_id(graph, "H3LipSyncMask"), 0])
        composite = by_class(graph, "ImageCompositeMasked")
        self.assertEqual(composite["mask"], [link_id(graph, "H3LipSyncMask"), 1])
        self.assertEqual(composite["destination"], mask["source_frames"])
        self.assertEqual(by_class(graph, "KSamplerSelect")["sampler_name"], "lcm")

    def test_prepared_crop_can_submit_with_fitted_audio(self):
        def stage(path, category):
            self.assertEqual(category, "lip_sync")
            if path.endswith(".wav"):
                with wave.open(path, "rb") as audio:
                    self.assertEqual((audio.getframerate(), audio.getnchannels()), (RATE, 1))
                    self.assertEqual(audio.getnframes(), round(22 * RATE / 24))
            return Path(path).name

        def submit(graph, client_id):
            self.assertTrue(by_class(graph, "SaveVideo")["filename_prefix"].endswith(client_id))
            return client_id

        services = SimpleNamespace(
            models=SimpleNamespace(load_model_config=lambda: self.models),
            media=SimpleNamespace(probe_video_metadata=Mock(return_value=SimpleNamespace(
                fps=24, width=640, height=640, frame_count=22)), stage_file=stage,
                resolve_output=lambda history, _: Path(f"{history}_00001_.mp4")),
            execution=SimpleNamespace(submit_prompt=Mock(side_effect=submit), wait_for_history=lambda prompt_id: prompt_id,
                                      poll_comfy_progress=lambda *_: []),
        )
        options = dict(dub=self.dub, speaker="A", first_frame=0, crop=Path("crop.mp4"), mask=Path("mask.mp4"),
                       services=services)
        result = list(render_lip_sync_preview(**options))
        self.assertIsInstance(result[-1], Path)

        services.execution.submit_prompt.reset_mock()
        preview = render_lip_sync_preview(**options)
        self.assertEqual(next(preview), "Checking the lip-sync window")
        self.assertEqual(next(preview), "Rendering the H3 mouth patch")
        with patch("h3_app.dubbing.lip_sync.check_cancelled", side_effect=JobCancelled("Stopped")):
            with self.assertRaises(JobCancelled):
                next(preview)
        services.execution.submit_prompt.assert_not_called()

    def test_invalid_crop_is_rejected_before_staging(self):
        media = SimpleNamespace(probe_video_metadata=Mock(return_value=SimpleNamespace(
            fps=24, width=641, height=640, frame_count=22)), stage_file=Mock())
        with self.assertRaisesRegex(H3Error, "multiples of 32"):
            list(render_lip_sync_preview(dub=self.dub, speaker="A", first_frame=0, crop=Path("crop.mp4"),
                                         mask=Path("mask.mp4"), services=SimpleNamespace(media=media)))
        media.stage_file.assert_not_called()

    def test_snap_h3_frames_rounds_up_to_17k_plus_5(self):
        self.assertEqual(snap_h3_frames(1), 5)
        self.assertEqual(snap_h3_frames(5), 5)
        self.assertEqual(snap_h3_frames(6), 22)
        self.assertEqual(snap_h3_frames(22), 22)
        self.assertEqual(snap_h3_frames(23), 39)
        self.assertEqual(snap_h3_frames(362), 362)
        with self.assertRaises(H3Error):
            snap_h3_frames(0)
        with self.assertRaises(H3Error):
            snap_h3_frames(363)

    def test_plan_windows_splits_cuts_skips_overlaps_and_uses_visual_hints(self):
        folder = player_folder(self.dub)
        samples = lambda start, end: round(end * RATE) - round(start * RATE)
        (folder / "alignment.json").write_text(json.dumps({
            "lines": [
                {"speaker": "A", "visual_speaker": {"time": 0.5, "visibility": "on_screen", "face_center": [0.25, 0.5]}},
                {"speaker": "A"},
                {"speaker": "B"},  # overlaps with line 1 from 2.2..2.6
                {"speaker": "A", "visual_speaker": {"time": 4.2, "visibility": "off_screen"}},
                {"speaker": "B", "visual_speaker": {"time": 5.5, "visibility": "on_screen", "face_center": [0.75, 0.4]}},
            ],
            "spoken": {
                "0": [0.2, 1.0],
                "1": [1.3, 2.6],
                "2": [2.2, 3.0],
                "3": [4.0, 4.6],
                "4": [5.0, 6.5],  # extends past 6.0s video_duration
            },
        }), encoding="utf-8")
        np.savez_compressed(folder / "alignment.npz", **{
            "0": np.full(samples(0.2, 1.0), 0.2, dtype=np.float32),
            "1": np.full(samples(1.3, 2.6), 0.2, dtype=np.float32),
            "2": np.full(samples(2.2, 3.0), 0.3, dtype=np.float32),
            "3": np.full(samples(4.0, 4.6), 0.2, dtype=np.float32),
            "4": np.full(samples(5.0, 6.5), 0.3, dtype=np.float32),
        })
        # Shot cut at 1.5s splits A's second line; overlap 2.2..2.6 is dropped; off_screen line 3 is skipped;
        # line 4 is clamped to 6.0s.
        windows = plan_lip_sync_windows(
            self.dub, video_duration=6.0, video_size=(1280, 720), shot_cuts=[1.5],
        )
        # Expect: A [0.2..1.5] (merged line 0 + pre-cut part of line 1), A [1.5..2.2], B [2.6..3.0], B [5.0..6.0]
        self.assertEqual([w.speaker for w in windows], ["A", "A", "B", "B"])
        # No window crosses the 1.5s cut.
        self.assertLessEqual(windows[0].end, 1.5 + 1e-6)
        self.assertGreaterEqual(windows[1].start, 1.5 - 1e-6)
        # Overlap 2.2..2.6 is excluded from active intervals.
        self.assertLessEqual(windows[1].active_end, 2.2 + 1e-6)
        self.assertGreaterEqual(windows[2].active_start, 2.6 - 1e-6)
        # Last window is clamped to video_duration=6.0.
        self.assertLessEqual(windows[-1].end, 6.0 + 1e-6)
        # Visual hints mapped to pixel coordinates.
        self.assertIsNotNone(windows[0].face_target)
        self.assertEqual(windows[0].face_target[:2], (320, 360))
        self.assertIsNotNone(windows[-1].face_target)
        self.assertEqual(windows[-1].face_target[:2], (960, 288))
        # All windows satisfy H3's 17k+5 render grid and never overlap each other.
        for prev, curr in zip(windows, windows[1:]):
            self.assertLessEqual(prev.end, curr.start + 1e-6)
        for w in windows:
            self.assertEqual((w.render_frames - 5) % 17, 0)
            self.assertGreaterEqual(w.render_frames, w.source_frames)

    def test_contained_overlap_stays_out_of_active_windows(self):
        data = {"lines": [{"speaker": "A"}, {"speaker": "B"}], "spoken": {"0": [0, 3], "1": [1, 1.5]}}
        (player_folder(self.dub) / "alignment.json").write_text(json.dumps(data), encoding="utf-8")
        for speaker in (None, "A"):
            windows = plan_lip_sync_windows(self.dub, 4, speaker=speaker)
            self.assertEqual([(w.active_start, w.active_end) for w in windows], [(0, 1), (1.5, 3)])

    def test_fitted_speech_across_cut_is_reviewable(self):
        from h3_app.dubbing.lip_sync_match import assign_speakers_to_shots

        lines = [{"speaker": "A", "start": 0.5, "end": 1}]
        spoken = {"0": [0.75, 1.5]}
        (player_folder(self.dub) / "alignment.json").write_text(json.dumps({"lines": lines, "spoken": spoken}))
        face = FaceTrack(3, 1, 1.25, 1.25, (TrackObservation(1.25, (0.3, 0.25, 0.2, 0.2)),))
        scan = SourceScan("test", 2, 640, 480, 24, (1,), (Shot(0, 0, 1), Shot(1, 1, 2, (face,))))
        assignments = assign_speakers_to_shots(scan, lines, spoken=spoken)
        self.assertEqual(assignments[(1, "A")].status, "uncertain")
        windows = plan_lip_sync_windows(self.dub, 2, shot_cuts=[1], assignments=assignments, require_assignment=True)
        self.assertEqual([w.shot_index for w in windows], [0])
        plan = build_lip_sync_plan_payload(dub=self.dub, scan=scan, lines=lines, spoken=spoken,
                                           assignments=assignments, windows=windows)
        revise_lip_sync_plan(self.dub, plan, {(1, "A"): 3}, reset=set())
        revised = load_lip_sync_plan(self.dub)
        self.assertEqual([w["shot_index"] for w in revised["windows"]], [0, 1])

    def test_manual_transcript_visual_speakers_flow_to_matching_and_windows(self):
        from h3_app.dubbing.lip_sync_match import Shot, SourceScan, assign_speakers_to_shots

        folder = player_folder(self.dub)
        samples = lambda start, end: round(end * RATE) - round(start * RATE)
        visual_speakers = [
            {"speaker": "A", "time": 0.5, "visibility": "on_screen", "face_center": [0.25, 0.5]},
            {"speaker": "B", "time": 2.5, "visibility": "off_screen"},
        ]
        lines = [
            {"speaker": "A", "start": 0.2, "end": 1.2},
            {"speaker": "B", "start": 2.0, "end": 3.0},
        ]
        (folder / "alignment.json").write_text(json.dumps({
            "lines": lines,
            "spoken": {"0": [0.2, 1.2], "1": [2.0, 3.0]},
            "visual_speakers": visual_speakers,
        }), encoding="utf-8")
        np.savez_compressed(folder / "alignment.npz", **{
            "0": np.full(samples(0.2, 1.2), 0.2, dtype=np.float32),
            "1": np.full(samples(2.0, 3.0), 0.3, dtype=np.float32),
        })
        scan = SourceScan(
            fingerprint="fp", duration=4.0, fps=24.0, width=1280, height=720, cuts=(), backend="none",
            shots=(Shot(index=0, start=0.0, end=4.0, tracks=()),),
        )
        assignments = assign_speakers_to_shots(scan, lines, visual_speakers=visual_speakers)
        self.assertEqual(assignments[(0, "A")].source, "gemini")
        self.assertEqual(assignments[(0, "A")].face_center, (0.25, 0.5))
        self.assertEqual(assignments[(0, "B")].status, "off_screen")
        windows = plan_lip_sync_windows(
            self.dub, video_duration=4.0, video_size=(1280, 720), assignments=assignments,
        )
        self.assertEqual([w.speaker for w in windows], ["A"])
        self.assertEqual(windows[0].face_target[:2], (320, 360))

    def test_single_gemini_hint_covers_later_windows_in_same_shot_without_tracks(self):
        from h3_app.dubbing.lip_sync_match import Shot, SourceScan, assign_speakers_to_shots

        folder = player_folder(self.dub)
        visual_speakers = [
            {"speaker": "A", "time": 0.5, "visibility": "on_screen", "face_center": [0.25, 0.5]},
        ]
        lines = [
            {"speaker": "A", "start": 0.2, "end": 1.2},
            {"speaker": "A", "start": 3.0, "end": 4.0},
        ]
        (folder / "alignment.json").write_text(json.dumps({
            "lines": lines,
            "spoken": {"0": [0.2, 1.2], "1": [3.0, 4.0]},
            "visual_speakers": visual_speakers,
        }), encoding="utf-8")
        np.savez_compressed(folder / "alignment.npz", **{"0": np.ones(1, dtype=np.float32), "1": np.ones(1, dtype=np.float32)})
        scan = SourceScan(
            fingerprint="fp", duration=5.0, fps=24.0, width=1280, height=720, cuts=(), backend="none",
            shots=(Shot(index=0, start=0.0, end=5.0, tracks=()),),
        )
        assignments = assign_speakers_to_shots(scan, lines, visual_speakers=visual_speakers)
        windows = plan_lip_sync_windows(
            self.dub, video_duration=5.0, video_size=(1280, 720), assignments=assignments, require_assignment=True,
        )
        self.assertEqual(len(windows), 2)
        self.assertEqual(windows[1].face_target[:2], (320, 360))
        self.assertEqual(windows[1].face_target[2], (windows[1].active_start_frame + windows[1].active_end_frame) // 2)

    def test_review_edit_overrides_off_screen_hint_and_preserves_other_decisions(self):
        folder = player_folder(self.dub)
        lines = [
            {"speaker": "A", "start": 0.3, "end": 1.0, "visual_speaker": {"visibility": "off_screen"}},
            {"speaker": "B", "start": 1.3, "end": 1.9},
        ]
        spoken = {"0": [0.3, 1.0], "1": [1.3, 1.9]}
        (folder / "alignment.json").write_text(json.dumps({"lines": lines, "spoken": spoken}), encoding="utf-8")
        np.savez(folder / "alignment.npz", **{key: np.ones(1) for key in spoken})
        face = FaceTrack(7, 0, 0.5, 1.6, (TrackObservation(0.5, (0.1, 0.1, 0.3, 0.3)),
                                          TrackObservation(1.6, (0.5, 0.1, 0.3, 0.3))))
        scan = SourceScan("fp", 2.0, 640, 480, 24.0, (), (Shot(0, 0.0, 2.0, (face,)),), "yolo-mediapipe")
        auto_decision = ShotSpeakerAssignment(0, "B", "matched", 7, (0.25, 0.25), 1.6, "gemini-decision", 0.9)
        assignments = {
            (0, "A"): ShotSpeakerAssignment(0, "A", "off_screen", None, None, 0.6, "gemini", 0.9),
            (0, "B"): auto_decision,
        }
        plan = build_lip_sync_plan_payload(
            dub=self.dub, scan=scan, lines=lines, spoken=spoken, assignments=assignments, windows=[],
        )
        revise_lip_sync_plan(self.dub, plan, {(0, "A"): 7}, reset=set())
        reviewed = load_lip_sync_plan(self.dub)
        self.assertNotEqual(reviewed["revision"], plan["revision"])
        self.assertEqual(reviewed["assignments"][0]["status"], "manual")
        self.assertEqual(reviewed["assignments"][1], auto_decision.to_dict())
        self.assertEqual([window["speaker"] for window in reviewed["windows"]], ["A", "B"])

        second = reviewed["windows"][1]
        self.assertEqual(second["face_target"], [416, 120, round(1.6 * 24) - second["first_frame"]])
        revise_lip_sync_plan(self.dub, reviewed, {}, reset={(0, "A")})
        self.assertEqual(load_lip_sync_plan(self.dub)["assignments"][0], assignments[(0, "A")].to_dict())

        # An empty reviewed selection must stay empty even if auto detection could now find a match.
        with patch("h3_app.dubbing.lip_sync.get_or_scan_source") as scanner:
            with self.assertRaisesRegex(H3Error, "No confident on-screen"):
                list(render_dub_lip_sync(dub=self.dub, services=None, reviewed_plan=plan))
            scanner.assert_not_called()

    def test_auto_graph_wires_face_track_lower_mask_audio_lock_and_stitch(self):
        graph = build_auto_lip_sync_graph(
            video="slice.mp4", dialogue="dialogue.wav", frames=73, models=self.models,
            refinement={}, output_prefix="auto", crop_size=640, active_start=4, active_end=60,
            face_target=(320, 240, 12),
        )
        track = by_class(graph, "H3FaceTrackCrop")
        self.assertEqual(track["select"], "closest_to_xy")
        self.assertEqual((track["X"], track["Y"], track["frame_index"]), (320, 240, 12))
        self.assertEqual(track["cut_detection"], "none")
        self.assertEqual(track["absent_shots"], "off")
        self.assertEqual(track["crop_factor"], 1.65)
        lower = by_class(graph, "H3LowerFaceMask")
        self.assertEqual(lower["crops"], [link_id(graph, "H3FaceTrackCrop"), 0])
        self.assertEqual(lower["transform"], [link_id(graph, "H3FaceTrackCrop"), 1])
        self.assertEqual((lower["active_start"], lower["active_end"]), (4, 60))
        prep = by_class(graph, "H3LipSyncPrepareCrop")
        self.assertEqual(prep["crops"], [link_id(graph, "H3FaceTrackCrop"), 0])
        self.assertEqual(prep["denoise_mask"], [link_id(graph, "H3LowerFaceMask"), 0])
        inject = by_class(graph, "H3InjectVideoLatent")
        self.assertEqual(inject["images"], [link_id(graph, "H3LipSyncPrepareCrop"), 0])
        mask = by_class(graph, "H3LipSyncMask")
        self.assertEqual(mask["mouth_mask"], [link_id(graph, "H3LowerFaceMask"), 0])
        blend = by_class(graph, "H3LipSyncBlend")
        self.assertEqual(blend["source_crops"], [link_id(graph, "H3FaceTrackCrop"), 0])
        self.assertEqual(blend["generated_crops"], [link_id(graph, "VAEDecode"), 0])
        self.assertTrue(lower["separate_opacity"])
        self.assertEqual(blend["composite_mask"], [link_id(graph, "H3LowerFaceMask"), 2])
        stitch = by_class(graph, "H3FaceStitch")
        self.assertEqual(stitch["masks"], [link_id(graph, "H3LowerFaceMask"), 1])
        self.assertEqual(stitch["refined_crops"], [link_id(graph, "H3LipSyncBlend"), 0])
        self.assertEqual(stitch["colour_match"], 0.0)
        self.assertEqual(stitch["base_images"], [link_id(graph, "GetVideoComponents"), 0])

        default_graph = build_auto_lip_sync_graph(
            video="slice.mp4", dialogue="dialogue.wav", frames=73, models=self.models,
            refinement={}, output_prefix="auto",
        )
        self.assertEqual(by_class(default_graph, "H3FaceTrackCrop")["select"], "largest_face")

    def test_auto_graph_uses_ui_default_singularity_profile_when_available(self):
        from h3_app.model_types import ModelConfig, ModelProfile

        prod_models = ModelConfig(
            {
                "speed": ModelProfile("Speed", "minimax_h3_fl2va_pruned_nvfp4.safetensors", "speed_ref.safetensors"),
                "singularity": ModelProfile(
                    "Singularity",
                    "Minimax-h3_Singularity_ref2va_Pruned_v1.3_int8.safetensors",
                    "Minimax-h3_Singularity_ref2va_Pruned_v1.3_int8.safetensors",
                ),
            },
            "speed", "text.safetensors", "video.safetensors", "audio.safetensors",
            turbo_lora="turbo.safetensors",
        )
        prod_models.turbo_lora_for = lambda _mode, _variant: "pika.safetensors"
        graph = build_auto_lip_sync_graph(
            video="slice.mp4", dialogue="dialogue.wav", frames=73, models=prod_models,
            refinement={"_source_duration": 73 / 24}, output_prefix="auto",
        )
        self.assertEqual(
            by_class(graph, "UNETLoader")["unet_name"],
            "Minimax-h3_Singularity_ref2va_Pruned_v1.3_int8.safetensors",
        )

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg/ffprobe required")
    def test_assembly_keeps_fractional_rate_frame_count_across_many_windows(self):
        root = self.dub.parent
        dub = root / "fractional.mp4"
        patch_video = root / "patch.mp4"
        for path, rate, frames in ((dub, "30000/1001", "300"), (patch_video, "24", "22")):
            subprocess.run([
                "ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", f"color=s=160x120:r={rate}",
                "-frames:v", frames, "-c:v", "libx264", str(path),
            ], check=True)
        windows = [
            (LipSyncWindow("A", index / 2, (index + 1) / 2, index / 2, (index + 1) / 2,
                           index * 12, 12, 22, 0, 12), patch_video)
            for index in range(20)
        ]
        # Include a source-only interval so the final timeline contains both kinds of segment.
        windows.pop(10)
        with patch("h3_app.dubbing.lip_sync.has_encoder", return_value=False):
            out = assemble_lip_sync_video(dub, windows, root / "assembled.mp4")
        self.assertEqual(probe_video_metadata(out).frame_count, 300)
        self.assertAlmostEqual(probe_video_metadata(out).fps, 30000 / 1001)

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg/ffprobe required")
    def test_render_dub_lip_sync_end_to_end_preserves_audio_and_subtitles(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            srt = root / "subs.srt"
            srt.write_text("1\n00:00:00,200 --> 00:00:01,500\nHello world\n", encoding="utf-8")
            dub = root / "dubbed.mp4"
            subprocess.run([
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-f", "lavfi", "-i", "color=c=navy:s=320x240:r=24:d=2.5",
                "-f", "lavfi", "-i", "sine=frequency=440:duration=2.5",
                "-i", str(srt),
                "-map", "0:v:0", "-map", "1:a:0", "-map", "1:a:0", "-map", "2:s:0",
                "-metadata:s:a:1", "language=eng",
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-c:s", "mov_text",
                str(dub),
            ], check=True)
            folder = player_folder(dub)
            folder.mkdir(parents=True)
            samples = round(1.2 * RATE) - round(0.4 * RATE)
            (folder / "alignment.json").write_text(json.dumps({
                "lines": [{"speaker": "A", "start": 0.4, "end": 1.2}],
                "spoken": {"0": [0.4, 1.2]},
            }), encoding="utf-8")
            np.savez_compressed(folder / "alignment.npz", **{"0": np.full(samples, 0.25, dtype=np.float32)})

            staged_videos: dict[str, Path] = {}
            out_dir = root / "comfy_out"
            out_dir.mkdir()

            def stage(path, category):
                self.assertEqual(category, "lip_sync")
                src = Path(path)
                if src.suffix == ".mp4":
                    copied = out_dir / f"staged_{len(staged_videos)}.mp4"
                    shutil.copy2(src, copied)
                    staged_videos[copied.name] = copied
                    return copied.name
                return src.name

            rendered_map: dict[str, Path] = {}

            def submit(graph, client_id):
                video_name = by_class(graph, "LoadVideo")["file"]
                expected_frames = by_class(graph, "MiniMaxH3ImageToVideo")["length"]
                slice_meta = probe_video_metadata(staged_videos[video_name])
                self.assertEqual(slice_meta.frame_count, expected_frames)
                self.assertEqual((slice_meta.frame_count - 5) % 17, 0)
                # Stub ComfyUI: return the sliced 24fps video as the rendered window output.
                rendered = out_dir / f"{client_id}_00001_.mp4"
                shutil.copy2(staged_videos[video_name], rendered)
                rendered_map[client_id] = rendered
                return client_id

            services = SimpleNamespace(
                models=SimpleNamespace(load_model_config=lambda: self.models),
                media=SimpleNamespace(
                    probe_video_metadata=probe_video_metadata,
                    stage_file=stage,
                    resolve_output=lambda history, _: rendered_map[history],
                ),
                execution=SimpleNamespace(
                    submit_prompt=Mock(side_effect=submit),
                    wait_for_history=lambda prompt_id: prompt_id,
                    poll_comfy_progress=lambda *_: [],
                ),
            )
            def scanner(_video, cuts):
                return SourceScan("test", 2.5, 320, 240, 24.0, tuple(cuts), (Shot(0, 0.0, 2.5),))

            plan = list(detect_lip_sync_plan(dub=dub, scanner=scanner))[-1]
            self.assertEqual(plan["windows_count"], 1)
            with patch("h3_app.dubbing.lip_sync.get_or_scan_source") as rescan:
                updates = list(render_dub_lip_sync(
                    dub=dub, services=services, crop_size=320, reviewed_plan=load_lip_sync_plan(dub),
                ))
                rescan.assert_not_called()
            assembled = updates[-1]
            self.assertIsInstance(assembled, Path)
            self.assertTrue(assembled.is_file())
            self.assertEqual(assembled.name, "dubbed.lipsync.mp4")

            # Verify output video dimensions, duration, audio stream, and subtitle stream are preserved.
            out_meta = probe_video_metadata(assembled)
            self.assertEqual((out_meta.width, out_meta.height), (320, 240))
            self.assertAlmostEqual(out_meta.duration, 2.5, delta=0.15)
            self.assertTrue(out_meta.has_audio)
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type:stream_tags=language:stream_disposition=default", "-of", "json", str(assembled)],
                capture_output=True, text=True, check=True,
            )
            stream_types = [s["codec_type"] for s in json.loads(probe.stdout)["streams"]]
            self.assertIn("video", stream_types)
            audio = [s for s in json.loads(probe.stdout)["streams"] if s["codec_type"] == "audio"]
            self.assertEqual(len(audio), 2)
            self.assertEqual(audio[0]["disposition"]["default"], 1)
            self.assertEqual(audio[1]["tags"]["language"], "eng")
            self.assertIn("subtitle", stream_types)

    def test_face_size_calculation_filtering_and_manual_override_exemption(self):
        folder = player_folder(self.dub)
        lines = [
            {"speaker": "A", "start": 0.1, "end": 0.6},
            {"speaker": "B", "start": 1.2, "end": 1.8},
        ]
        spoken = {"0": [0.1, 0.6], "1": [1.2, 1.8]}
        (folder / "alignment.json").write_text(json.dumps({"lines": lines, "spoken": spoken}), encoding="utf-8")
        np.savez_compressed(folder / "alignment.npz", **{
            "0": np.ones(round(0.5 * RATE), dtype=np.float32),
            "1": np.ones(round(0.6 * RATE), dtype=np.float32),
        })
        small_track = FaceTrack(0, 0, 0.1, 0.9, (
            TrackObservation(0.15, (0.1, 0.1, 0.04, 0.04)),
            TrackObservation(0.35, (0.1, 0.1, 0.06, 0.06)),
            TrackObservation(0.55, (0.1, 0.1, 0.20, 0.20)),
            TrackObservation(0.90, (0.1, 0.1, 0.20, 0.20)),
        ))
        large_track = FaceTrack(1, 1, 1.2, 1.8, (
            TrackObservation(1.3, (0.3, 0.2, 0.25, 0.30)),
            TrackObservation(1.6, (0.3, 0.2, 0.25, 0.34)),
        ))
        scan = SourceScan("fp", 2.0, 1280, 720, 24.0, (1.0,), (
            Shot(0, 0.0, 1.0, (small_track,)),
            Shot(1, 1.0, 2.0, (large_track,)),
        ), "yolo-mediapipe")
        assignments = {
            (0, "A"): ShotSpeakerAssignment(0, "A", "matched", 0, (0.13, 0.13), 0.35, "yolo-mediapipe", 0.9),
            (1, "B"): ShotSpeakerAssignment(1, "B", "matched", 1, (0.425, 0.36), 1.5, "yolo-mediapipe", 0.9),
        }
        tracks = {(0, 0): small_track, (1, 1): large_track}
        windows = plan_lip_sync_windows(
            self.dub, 2.0, video_size=(1280, 720), shot_cuts=[1.0],
            assignments=assignments, tracks=tracks, require_assignment=True,
        )
        self.assertEqual([w.face_size for w in windows], [0.06, 0.34])
        self.assertEqual([w.manual for w in windows], [False, False])

        # Range filter excludes the small auto face and keeps the larger one.
        filtered = filter_lip_sync_windows(windows, min_face_size=0.10, max_face_size=0.80)
        self.assertEqual([w.speaker for w in filtered], ["B"])

        # Unmeasured face_size (None) is never dropped.
        unmeasured = LipSyncWindow("C", 0.0, 0.5, 0.0, 0.5, 0, 12, 22, 0, 12, None, 0, None, False)
        self.assertEqual(filter_lip_sync_windows([unmeasured], min_face_size=0.2, max_face_size=0.4), [unmeasured])

        # Manual override on shot 0 keeps the small face exempt from size filtering, and reset restores auto.
        plan = build_lip_sync_plan_payload(
            dub=self.dub, scan=scan, lines=lines, spoken=spoken, assignments=assignments, windows=windows,
        )
        self.assertEqual(plan["window_version"], 3)
        legacy_windows = [
            {key: value for key, value in window.items() if key not in {"face_size", "manual"}}
            for window in plan["windows"]
        ]
        (folder / PLAN_FILE).write_text(
            json.dumps({**plan, "window_version": 2, "windows": legacy_windows}), encoding="utf-8",
        )
        upgraded = load_lip_sync_plan(self.dub)
        self.assertEqual(upgraded["window_version"], 3)
        self.assertEqual(upgraded["revision"], plan["revision"])
        self.assertEqual([w["face_size"] for w in upgraded["windows"]], [0.06, 0.34])
        revise_lip_sync_plan(self.dub, plan, {(0, "A"): 0}, reset=set())
        revised = load_lip_sync_plan(self.dub)
        self.assertTrue(revised["windows"][0]["manual"])
        self.assertFalse(revised["windows"][1]["manual"])
        revised_windows = [LipSyncWindow(**w) for w in revised["windows"]]
        self.assertEqual(
            [w.speaker for w in filter_lip_sync_windows(revised_windows, min_face_size=0.10, max_face_size=0.25)],
            ["A"],
        )
        revise_lip_sync_plan(self.dub, revised, {}, reset={(0, "A")})
        restored = load_lip_sync_plan(self.dub)
        self.assertFalse(restored["windows"][0]["manual"])

        for bad_min, bad_max in ((0.6, 0.4), (-0.1, 0.5), (0.0, 1.1), (True, 1.0), (float("nan"), 1.0)):
            with self.assertRaises(H3Error):
                validate_face_size_range(bad_min, bad_max)

    def test_render_dub_lip_sync_filters_by_face_size_and_keeps_manual_override(self):
        folder = player_folder(self.dub)
        lines = [
            {"speaker": "A", "start": 0.1, "end": 0.6},
            {"speaker": "B", "start": 1.2, "end": 1.8},
        ]
        spoken = {"0": [0.1, 0.6], "1": [1.2, 1.8]}
        (folder / "alignment.json").write_text(json.dumps({"lines": lines, "spoken": spoken}), encoding="utf-8")
        np.savez_compressed(folder / "alignment.npz", **{
            "0": np.ones(round(0.5 * RATE), dtype=np.float32),
            "1": np.ones(round(0.6 * RATE), dtype=np.float32),
        })
        small_track = FaceTrack(0, 0, 0.1, 0.6, (TrackObservation(0.35, (0.1, 0.1, 0.05, 0.05)),))
        large_track = FaceTrack(1, 1, 1.2, 1.8, (TrackObservation(1.5, (0.3, 0.2, 0.25, 0.30)),))
        scan = SourceScan("fp", 2.0, 1280, 720, 24.0, (1.0,), (
            Shot(0, 0.0, 1.0, (small_track,)),
            Shot(1, 1.0, 2.0, (large_track,)),
        ), "yolo-mediapipe")
        assignments = {
            (0, "A"): ShotSpeakerAssignment(0, "A", "matched", 0, (0.125, 0.125), 0.35, "yolo-mediapipe", 0.9),
            (1, "B"): ShotSpeakerAssignment(1, "B", "matched", 1, (0.425, 0.35), 1.5, "yolo-mediapipe", 0.9),
        }
        windows = plan_lip_sync_windows(
            self.dub, 2.0, video_size=(1280, 720), shot_cuts=[1.0],
            assignments=assignments, tracks={(0, 0): small_track, (1, 1): large_track}, require_assignment=True,
        )
        plan = build_lip_sync_plan_payload(
            dub=self.dub, scan=scan, lines=lines, spoken=spoken, assignments=assignments, windows=windows,
        )

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            rendered_files: dict[str, Path] = {}
            submitted_shots: list[tuple[int, int, int] | None] = []

            def submit(graph, client_id):
                track = by_class(graph, "H3FaceTrackCrop")
                submitted_shots.append((track["X"], track["Y"], track["frame_index"]))
                out = out_dir / f"{client_id}_00001_.mp4"
                out.write_bytes(b"rendered")
                rendered_files[client_id] = out
                return client_id

            def fake_slice(_source, _window, destination, **_kw):
                destination.write_bytes(b"slice")
                return destination

            def fake_assemble(_dub, rendered, destination, **_kw):
                destination.write_bytes(b"assembled")
                return destination

            services = SimpleNamespace(
                models=SimpleNamespace(load_model_config=lambda: self.models),
                media=SimpleNamespace(
                    stage_file=lambda p, _cat: Path(p).name,
                    resolve_output=lambda h, _: rendered_files[h],
                ),
                execution=SimpleNamespace(
                    submit_prompt=Mock(side_effect=submit),
                    wait_for_history=lambda prompt_id: prompt_id,
                    poll_comfy_progress=lambda *_: [],
                ),
            )
            with patch("h3_app.dubbing.lip_sync._extract_window_slice", side_effect=fake_slice), \
                    patch("h3_app.dubbing.lip_sync.assemble_lip_sync_video", side_effect=fake_assemble):
                # Filter out the 5% auto face, rendering only the 30% face.
                result = list(render_dub_lip_sync(
                    dub=self.dub, services=services, reviewed_plan=plan, use_cache=False,
                    min_face_size=0.10, max_face_size=0.80,
                ))
                self.assertEqual(len(submitted_shots), 1)
                self.assertEqual(submitted_shots[0][:2], (544, 252))
                self.assertEqual(result[-1], self.dub.with_name("finished.lipsync.mp4"))
                report = json.loads((folder / REPORT_FILE).read_text(encoding="utf-8"))
                self.assertEqual(report["total_windows"], 1)
                self.assertEqual(report["filtered_windows"], 1)
                self.assertEqual(report["face_size_range"], [0.1, 0.8])

                # Reject when every window is filtered out.
                with self.assertRaisesRegex(H3Error, "face size range"):
                    list(render_dub_lip_sync(
                        dub=self.dub, services=services, reviewed_plan=plan, use_cache=False,
                        min_face_size=0.50, max_face_size=0.90,
                    ))

                # Manual selection on shot 0 (5%) survives even when range excludes both 5% and 30%.
                submitted_shots.clear()
                revise_lip_sync_plan(self.dub, plan, {(0, "A"): 0}, reset=set())
                reviewed = load_lip_sync_plan(self.dub)
                list(render_dub_lip_sync(
                    dub=self.dub, services=services, reviewed_plan=reviewed, use_cache=False,
                    min_face_size=0.10, max_face_size=0.20,
                ))
                self.assertEqual(len(submitted_shots), 1)
                self.assertEqual(submitted_shots[0][:2], (160, 90))
                report2 = json.loads((folder / REPORT_FILE).read_text(encoding="utf-8"))
                self.assertEqual(report2["total_windows"], 1)
                self.assertEqual(report2["filtered_windows"], 1)

    def test_validate_engine_and_cache_key_isolation(self):
        self.assertEqual(validate_lip_sync_engine(None), "h3")
        self.assertEqual(validate_lip_sync_engine(" MuseTalk "), "musetalk")
        self.assertEqual(validate_lip_sync_engine("LATENTSYNC"), "latentsync")
        with self.assertRaisesRegex(H3Error, "Unsupported lip-sync engine"):
            validate_lip_sync_engine("ltx25")

        win = LipSyncWindow("A", 0.0, 0.5, 0.0, 0.5, 0, 12, 22, 0, 12, (320, 240, 6), 0, 0.25, False)
        audio = np.ones(12000, dtype=np.float32)
        common = dict(crop_size=640, strength=0.7, seed=0, refinement={"_feather": 6})
        k_h3 = window_cache_key(self.dub, win, audio, engine="h3", **common)
        k_mt_0 = window_cache_key(self.dub, win, audio, engine="musetalk", engine_options={"bbox_shift": 0}, **common)
        k_mt_5 = window_cache_key(self.dub, win, audio, engine="musetalk", engine_options={"bbox_shift": 5}, **common)
        k_ls_20 = window_cache_key(
            self.dub, win, audio, engine="latentsync", engine_options={"inference_steps": 20, "guidance_scale": 1.5}, **common
        )
        k_ls_30 = window_cache_key(
            self.dub, win, audio, engine="latentsync", engine_options={"inference_steps": 30, "guidance_scale": 1.5}, **common
        )
        self.assertEqual(len({k_h3, k_mt_0, k_mt_5, k_ls_20, k_ls_30}), 5)

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg/ffprobe required")
    def test_extract_window_slice_pad_false_keeps_exact_source_frames(self):
        root = self.dub.parent
        src = root / "src.mp4"
        subprocess.run([
            "ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "color=s=160x120:r=24:d=2.0",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", str(src),
        ], check=True)
        win = LipSyncWindow("A", 0.0, 0.5, 0.0, 0.5, 0, 12, 22, 0, 12)
        unpadded = _extract_window_slice(src, win, root / "unpadded.mp4", pad=False)
        padded = _extract_window_slice(src, win, root / "padded.mp4", pad=True)
        self.assertEqual(probe_video_metadata(unpadded).frame_count, 12)
        self.assertEqual(probe_video_metadata(padded).frame_count, 22)

    def test_render_dub_lip_sync_dispatches_musetalk_and_latentsync_with_cache(self):
        folder = player_folder(self.dub)
        lines = [{"speaker": "A", "start": 0.1, "end": 0.6}]
        spoken = {"0": [0.1, 0.6]}
        (folder / "alignment.json").write_text(json.dumps({"lines": lines, "spoken": spoken}), encoding="utf-8")
        np.savez_compressed(folder / "alignment.npz", **{"0": np.ones(round(0.5 * RATE), dtype=np.float32)})
        track = FaceTrack(0, 0, 0.1, 0.6, (TrackObservation(0.35, (0.2, 0.2, 0.3, 0.3)),))
        scan = SourceScan("fp", 1.5, 640, 480, 24.0, (), (Shot(0, 0.0, 1.5, (track,)),), "yolo-mediapipe")
        assignments = {
            (0, "A"): ShotSpeakerAssignment(0, "A", "matched", 0, (0.35, 0.35), 0.35, "yolo-mediapipe", 0.9),
        }
        windows = plan_lip_sync_windows(
            self.dub, 1.5, video_size=(640, 480), assignments=assignments, tracks={(0, 0): track}, require_assignment=True,
        )
        plan = build_lip_sync_plan_payload(
            dub=self.dub, scan=scan, lines=lines, spoken=spoken, assignments=assignments, windows=windows,
        )
        runtime = SimpleNamespace(output_dir=self.dub.parent, comfy_dir=self.dub.parent, script_dir=self.dub.parent)
        free_gpu = Mock()
        calls: list[tuple[str, dict]] = []
        pad_flags: list[bool] = []

        def fake_slice(_source, _window, destination, *, pad=True, **_kw):
            pad_flags.append(pad)
            destination.write_bytes(b"slice")
            return destination

        def fake_worker_run(op, payload, rt, *, free_gpu, hf_token=None, **_kw):
            calls.append((op, payload))
            self.assertIs(rt, runtime)
            self.assertIsNotNone(free_gpu)
            self.assertEqual(hf_token, "hf-secret")
            out = Path(payload["output_path"])
            out.write_bytes(f"{op}-rendered".encode("utf-8"))
            return {"output_path": str(out), "frames": payload["source_frames"]}

        def fake_assemble(_dub, rendered, destination, **_kw):
            destination.write_bytes(b"assembled")
            return destination

        with patch("h3_app.dubbing.lip_sync._extract_window_slice", side_effect=fake_slice), \
                patch("h3_app.dubbing.lip_sync.assemble_lip_sync_video", side_effect=fake_assemble), \
                patch("h3_app.dubbing.lip_sync.lip_sync_runtime.run", side_effect=fake_worker_run):
            with self.assertRaisesRegex(H3Error, "runtime is not configured"):
                list(render_dub_lip_sync(dub=self.dub, services=None, reviewed_plan=plan, engine="musetalk"))

            list(render_dub_lip_sync(
                dub=self.dub, services=None, reviewed_plan=plan, engine="musetalk",
                bbox_shift=4, feather=8, runtime=runtime, free_gpu=free_gpu, hf_token="hf-secret",
            ))
            self.assertEqual(pad_flags, [False])
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0][0], "render_musetalk")
            self.assertEqual(calls[0][1]["bbox_shift"], 4)
            self.assertEqual(calls[0][1]["feather"], 8)
            self.assertEqual(calls[0][1]["source_frames"], windows[0].source_frames)
            self.assertEqual(calls[0][1]["face_target"], list(windows[0].face_target))
            report = json.loads((folder / REPORT_FILE).read_text(encoding="utf-8"))
            self.assertEqual(report["engine"], "musetalk")

            # Second MuseTalk run with identical settings reuses cache without invoking worker.
            list(render_dub_lip_sync(
                dub=self.dub, services=None, reviewed_plan=plan, engine="musetalk",
                bbox_shift=4, feather=8, runtime=runtime, free_gpu=free_gpu, hf_token="hf-secret",
            ))
            self.assertEqual(len(calls), 1)
            self.assertEqual(json.loads((folder / REPORT_FILE).read_text(encoding="utf-8"))["cached_windows"], 1)

            # Switching to LatentSync misses MuseTalk cache and passes diffusion parameters.
            list(render_dub_lip_sync(
                dub=self.dub, services=None, reviewed_plan=plan, engine="latentsync",
                inference_steps=30, guidance_scale=2.0, feather=10, runtime=runtime, free_gpu=free_gpu, hf_token="hf-secret",
            ))
            self.assertEqual(len(calls), 2)
            self.assertEqual(calls[1][0], "render_latentsync")
            self.assertEqual(calls[1][1]["inference_steps"], 30)
            self.assertEqual(calls[1][1]["guidance_scale"], 2.0)
            self.assertEqual(calls[1][1]["feather"], 10)
            self.assertEqual(json.loads((folder / REPORT_FILE).read_text(encoding="utf-8"))["engine"], "latentsync")

            # No-face error skips gracefully and fails when zero windows render.
            with patch("h3_app.dubbing.lip_sync.lip_sync_runtime.run", side_effect=RuntimeError("No detected face")):
                with self.assertRaisesRegex(H3Error, "No faces were detected"):
                    list(render_dub_lip_sync(
                        dub=self.dub, services=None, reviewed_plan=plan, engine="musetalk",
                        use_cache=False, runtime=runtime, free_gpu=free_gpu,
                    ))

    def test_lip_sync_worker_pure_helpers_and_serve_protocol(self):
        # select_target_face anchors on face_target and tracks across frames even with another face.
        detections = [
            [(10, 10, 110, 110), (400, 100, 500, 200)],
            [(12, 12, 112, 112), (404, 102, 504, 202)],
            [],  # brief occlusion
            [(408, 104, 508, 204), (14, 14, 114, 114)],
        ]
        tracked = lip_sync_worker.select_target_face(detections, (452, 152, 1))
        self.assertEqual(tracked[0], (400, 100, 500, 200))
        self.assertEqual(tracked[1], (404, 102, 504, 202))
        self.assertIsNone(tracked[2])
        self.assertEqual(tracked[3], (408, 104, 508, 204))
        self.assertEqual(lip_sync_worker.select_target_face([[], []], (100, 100, 0)), [None, None])

        # Largest face chosen when face_target is None.
        largest = lip_sync_worker.select_target_face(
            [[(0, 0, 40, 40), (200, 200, 360, 360)], [(0, 0, 42, 42), (204, 204, 364, 364)]], None
        )
        self.assertEqual(largest, [(200, 200, 360, 360), (204, 204, 364, 364)])

        # temporal_opacity_weights ramps at padded boundaries and keeps full-span active frames at 1.0.
        padded_weights = lip_sync_worker.temporal_opacity_weights(12, 3, 9, ramp_frames=3)
        self.assertEqual(padded_weights[:3], [0.0, 0.0, 0.0])
        self.assertEqual(padded_weights[9:], [0.0, 0.0, 0.0])
        self.assertAlmostEqual(padded_weights[3], 1 / 3, places=4)
        self.assertAlmostEqual(padded_weights[4], 2 / 3, places=4)
        self.assertEqual(padded_weights[5:7], [1.0, 1.0])
        self.assertAlmostEqual(padded_weights[7], 2 / 3, places=4)
        self.assertAlmostEqual(padded_weights[8], 1 / 3, places=4)
        self.assertEqual(lip_sync_worker.temporal_opacity_weights(5, 0, -1, ramp_frames=3), [1.0] * 5)

        # resample_frame_indices maps endpoints and preserves monotonicity.
        idx_up = lip_sync_worker.resample_frame_indices(24, 25)
        self.assertEqual((len(idx_up), idx_up[0], idx_up[-1]), (25, 0, 23))
        self.assertEqual(idx_up, sorted(idx_up))
        self.assertEqual(lip_sync_worker.resample_frame_indices(10, 10), list(range(10)))

        # serve() JSON-line loop handles progress, ok replies, and errors.
        requests = [
            json.dumps({"id": 1, "op": "render_musetalk", "progress": True}) + "\n",
            json.dumps({"id": 2, "op": "unload"}) + "\n",
            json.dumps({"id": 3, "op": "bogus"}) + "\n",
        ]
        replies = io.StringIO()

        def fake_handle(worker_self, req):
            if req["op"] == "render_musetalk":
                worker_self.progress("Synthesizing")
                worker_self._models["musetalk_v15"] = object()
                return {"output_path": "out.mp4", "frames": 12}
            if req["op"] == "unload":
                return worker_self._unload(req.get("model"))
            raise ValueError("bad op")

        with patch.object(lip_sync_worker.Worker, "handle", fake_handle):
            lip_sync_worker.serve(str(self.dub.parent), requests=requests, replies=replies)
        lines = [json.loads(line) for line in replies.getvalue().splitlines()]
        self.assertEqual(lines[0], {"id": 1, "event": "progress", "status": "Synthesizing"})
        self.assertTrue(lines[1]["ok"])
        self.assertEqual(lines[1]["loaded"], ["musetalk_v15"])
        self.assertTrue(lines[2]["ok"])
        self.assertEqual(lines[2]["loaded"], [])
        self.assertFalse(lines[3]["ok"])
        self.assertIn("ValueError: bad op", lines[3]["error"])
