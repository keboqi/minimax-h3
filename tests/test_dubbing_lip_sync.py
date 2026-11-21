"""Frozen dialogue and graph wiring for a prepared H3 lip-sync shot."""

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
    assemble_lip_sync_video,
    fitted_dialogue,
    plan_lip_sync_windows,
    render_dub_lip_sync,
    render_lip_sync_preview,
)
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
        lower = by_class(graph, "H3LowerFaceMask")
        self.assertEqual(lower["crops"], [link_id(graph, "H3FaceTrackCrop"), 0])
        self.assertEqual(lower["transform"], [link_id(graph, "H3FaceTrackCrop"), 1])
        self.assertEqual((lower["active_start"], lower["active_end"]), (4, 60))
        mask = by_class(graph, "H3LipSyncMask")
        self.assertEqual(mask["mouth_mask"], [link_id(graph, "H3LowerFaceMask"), 0])
        stitch = by_class(graph, "H3FaceStitch")
        self.assertEqual(stitch["masks"], [link_id(graph, "H3LowerFaceMask"), 1])
        self.assertEqual(stitch["refined_crops"], [link_id(graph, "VAEDecode"), 0])
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
                "-map", "0:v:0", "-map", "1:a:0", "-map", "2:s:0",
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-c:s", "mov_text",
                str(dub),
            ], check=True)
            folder = player_folder(dub)
            folder.mkdir(parents=True)
            samples = round(1.2 * RATE) - round(0.4 * RATE)
            (folder / "alignment.json").write_text(json.dumps({
                "lines": [{"speaker": "A"}],
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
            updates = list(render_dub_lip_sync(dub=dub, services=services, crop_size=320))
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
                ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type", "-of", "json", str(assembled)],
                capture_output=True, text=True, check=True,
            )
            stream_types = [s["codec_type"] for s in json.loads(probe.stdout)["streams"]]
            self.assertIn("video", stream_types)
            self.assertIn("audio", stream_types)
            self.assertIn("subtitle", stream_types)
