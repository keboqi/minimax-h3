"""Gemini-only multimodal take review (review + finetuned prompt) and Gemini Decision take scoring
for MiniMax H3 and Storyboard. No network requests leave the machine."""

from __future__ import annotations

import json
import os
from pathlib import Path
import struct
from tempfile import TemporaryDirectory
import time
import unittest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from h3_app import media_review, storyboard as sb, take_review
from h3_app.catalog import DEFAULT_GEMINI_PROMPT_MODEL, UI_DEFAULTS
from h3_app.config import RuntimeConfig
from h3_app.contracts import GENERATION_FIELDS
from h3_app.errors import H3Error
from h3_app.generation.results import GenerationUpdate
from h3_app.jobs import JOBS
from h3_app.llm import decisions, gemini
from h3_app.workspace_store import WorkspaceStore, install_owner_cookie
from h3_ui.h3.handlers import H3Handlers, H3Services
from h3_ui.h3.page import H3PageServices, install_h3_page
from h3_ui.job_bindings import owned_generation
from h3_ui.storyboard.handlers import StoryboardHandlers, StoryboardServices
from h3_ui.storyboard.page import install_storyboard_page

REPO_ROOT = Path(__file__).resolve().parents[1]
PROFILE_VALUES = {name: UI_DEFAULTS.get(name, f"default-{name}") for name in GENERATION_FIELDS}


def _fake_decision_answers(questions, blocks, *, key, **kwargs):
    answers = {}
    for name, question in questions.items():
        if question["type"] == "score":
            score = {
                "fit": 3.2,
                "reference_fidelity": 2.8,
                "visual_quality": 3.6,
                "audio_quality": 3.0,
                "advice_fit": 2.4,
                "overall": 3.2,
            }.get(name, 3.0)
            answers[name] = {
                "type": "score",
                "score": score,
                "legend": {str(i): text for i, text in enumerate(question["criteria"])},
            }
        elif question["type"] == "choice":
            first = next(iter(question["criteria"]))
            answers[name] = {"type": "choice", "choice": first, "probabilities": {first: 0.85}}
        elif question["type"] == "noul":
            answers[name] = {"type": "noul", "noul": 0.85 if name == "tag_speech" else 0.1}
    return answers


def _fake_gemini_review_payload(finetuned="integrated_multimodal_description: [Shot 1] Refined.\n\noverall_soundscape: Calm wind.\n\nnon_diegetic_music: N/A"):
    body = {
        "summary": "Good opening framing, but lip-sync drifts slightly and motion is rushed.",
        "strengths": ["Matches the first frame lighting and wardrobe."],
        "issues": [
            {"area": "audio", "severity": "major", "timestamp": "00:02.0-00:03.2", "detail": "Speech trails behind lip motion."},
            {"area": "reference", "severity": "minor", "timestamp": "", "detail": "Jacket collar shifts slightly at the end."},
        ],
        "prompt_changes": ["Slowed the head turn so dialogue fits comfortably.", "Anchored the jacket collar to <Picture 1>."],
        "finetuned_prompt": finetuned,
    }
    return {"candidates": [{"content": {"parts": [{"text": json.dumps(body)}]}}]}


class _FakeGeminiSession:
    """In-memory stand-in for ``gemini.Session`` that records uploaded files and generate calls."""

    def __init__(self, key: str, *, reply_payload: dict | None = None, recorder: dict | None = None):
        self.key = key
        self.reply_payload = reply_payload or _fake_gemini_review_payload()
        self.recorder = recorder if recorder is not None else {}
        self.recorder.setdefault("uploads", [])
        self.recorder.setdefault("generates", [])
        self.recorder["key"] = key

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.recorder["closed"] = True

    def upload(self, path: Path) -> dict:
        self.recorder["uploads"].append(Path(path))
        return {"name": f"files/{len(self.recorder['uploads'])}", "uri": f"https://generativelanguage.googleapis.com/v1beta/files/{Path(path).name}",
                "mimeType": decisions.media_type(Path(path)), "state": "ACTIVE"}

    def wait_until_active(self, info: dict) -> dict:
        return info

    def file_part(self, path: Path) -> dict:
        info = self.wait_until_active(self.upload(path))
        return {"fileData": {"mimeType": info["mimeType"], "fileUri": info["uri"]}}

    def generate(self, model: str, parts: list[dict], *, generation_config: dict, system_prompt: str | None = None, **kwargs) -> dict:
        self.recorder["generates"].append({
            "model": model, "parts": parts, "generation_config": generation_config,
            "system_prompt": system_prompt, "kwargs": kwargs,
        })
        return self.reply_payload


class TakeReviewCoreTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.runtime = RuntimeConfig(
            script_dir=REPO_ROOT,
            comfy_url="http://comfy",
            comfy_dir=self.root / "comfy",
            models_config=self.root / "models.json",
            outputs_dir=self.root / "out",
        )
        self.take = self.root / "take.mp4"
        self.take.write_bytes(b"fake-mp4-video-audio")
        self.pic = self.root / "hero.png"
        Image.new("RGB", (128, 128), (40, 120, 90)).save(self.pic)
        self.voice = self.root / "voice.wav"
        self.voice.write_bytes(b"RIFFfakewav")

    def test_collect_h3_references_includes_fl2va_voices_and_ref_media(self):
        fl2va_refs = take_review.collect_h3_references(
            "First / last frame",
            first_image=self.pic,
            last_image=self.pic,
            fl2va_audios=[self.voice, None, None],
        )
        self.assertEqual([label for label, _ in fl2va_refs], [
            "<Picture 1> (first frame)",
            "<Picture 2> (last frame)",
            "<Audio 1> (voice reference)",
        ])
        ref_refs = take_review.collect_h3_references(
            "Reference media",
            ref_images=[self.pic],
            ref_videos=[self.take],
            ref_audios=[self.voice],
            cast_names=["Mei"],
        )
        self.assertEqual([label for label, _ in ref_refs], [
            "<Picture 1> (cast: Mei)",
            "<Video 1>",
            "<Audio 1>",
        ])

    def test_decision_scoring_uses_take_prompt_refs_and_advice_and_preserves_llm(self):
        take_input = take_review.TakeReviewInput(
            take_path=self.take,
            prompt="Mei speaks softly in <Audio 1>.",
            mode="First / last frame",
            duration=6.0,
            width=1280,
            height=720,
            references=(("<Picture 1> (first frame)", self.pic), ("<Audio 1> (voice reference)", self.voice)),
            advice="Make her lip-sync crisp and keep the lighting warm.",
        )
        media_review.save_llm(self.take, {"summary": "Prior LLM review", "finetuned_prompt": "Old refined prompt"})

        with patch.dict(os.environ, {"H3_DECISION": "1"}), \
                patch.object(decisions, "decide", side_effect=_fake_decision_answers) as asked:
            scored = take_review.score_take(take_input, api_key="gem-key")

        self.assertEqual(asked.call_count, 1)
        questions, blocks = asked.call_args.args
        self.assertTrue({"fit", "reference_fidelity", "visual_quality", "audio_quality", "advice_fit", "overall"} <= set(questions))
        media_types = [block["type"] for block in blocks if block.get("type") != "text"]
        self.assertEqual(media_types, ["image", "audio", "video"])
        combined_text = "\n".join(block["text"] for block in blocks if block.get("type") == "text")
        self.assertIn("Mei speaks softly", combined_text)
        self.assertIn("Make her lip-sync crisp", combined_text)
        self.assertEqual(scored["score"], 80)
        self.assertEqual(scored["scores"]["references"], 70)
        self.assertEqual(scored["scores"]["advice"], 60)
        self.assertEqual(scored["ref_count"], 2)
        self.assertEqual(scored["advice"], "Make her lip-sync crisp and keep the lighting warm.")
        self.assertIn("dialogue", scored["tags"])
        # Saving Decision score must keep the existing persisted LLM review intact.
        self.assertEqual(media_review.public_llm(self.take)["summary"], "Prior LLM review")

    def test_decision_scoring_falls_back_to_files_api_uri_when_inline_budget_exceeded(self):
        recorder = {}
        take_input = take_review.TakeReviewInput(
            take_path=self.take,
            prompt="A street chase.",
            mode="Reference media",
            references=(("<Picture 1>", self.pic),),
        )
        with patch.dict(os.environ, {"H3_DECISION": "1"}), \
                patch.object(decisions, "MAX_INLINE_BYTES", 1), \
                patch.object(gemini, "Session", side_effect=lambda k: _FakeGeminiSession(k, recorder=recorder)), \
                patch.object(decisions, "decide", side_effect=_fake_decision_answers) as asked:
            scored = take_review.score_take(take_input, api_key="gem-key")
        self.assertEqual(scored["status"], "done")
        blocks = asked.call_args.args[1]
        media_blocks = [block for block in blocks if block.get("type") in ("image", "video", "audio")]
        self.assertTrue(all("uri" in block and "data" not in block for block in media_blocks))
        self.assertTrue(recorder.get("closed"))

    def test_gemini_llm_review_sends_video_refs_prompt_and_advice_and_preserves_score(self):
        recorder = {}
        media_review.save_decision(self.take, {"status": "done", "fit": 75, "label": "Good fit", "tags": ["city"]})
        take_input = take_review.TakeReviewInput(
            take_path=self.take,
            prompt="Mei smiles and says hello using <Audio 1>.",
            mode="First / last frame",
            duration=5.0,
            width=1280,
            height=720,
            references=(("<Picture 1> (first frame)", self.pic), ("<Audio 1> (voice reference)", self.voice)),
            advice="Keep the camera completely still.",
        )
        with patch.object(gemini, "Session", side_effect=lambda k: _FakeGeminiSession(k, recorder=recorder)):
            reviewed = take_review.review_take_with_gemini(
                take_input,
                model="gemini-3.8-flash",
                api_key="gem-key",
                runtime=self.runtime,
            )
        self.assertEqual(reviewed["model"], "gemini-3.8-flash")
        self.assertEqual(reviewed["ref_count"], 2)
        self.assertEqual(reviewed["advice"], "Keep the camera completely still.")
        self.assertEqual(len(reviewed["issues"]), 2)
        self.assertEqual(reviewed["issues"][0]["area"], "audio")
        self.assertIn("integrated_multimodal_description:", reviewed["finetuned_prompt"])
        # Uploads: hero.png, voice.wav, take.mp4 (and Session closed cleanly).
        self.assertEqual([path.name for path in recorder["uploads"]], ["hero.png", "voice.wav", "take.mp4"])
        self.assertTrue(recorder.get("closed"))
        call = recorder["generates"][0]
        self.assertEqual(call["generation_config"]["responseMimeType"], "application/json")
        self.assertIn("MiniMax H3", call["system_prompt"])
        self.assertIn("Keep the camera completely still.", call["parts"][0]["text"])
        # Persisted LLM review coexists with prior Decision score.
        self.assertEqual(media_review.public(self.take)["fit"], 75)
        self.assertEqual(media_review.public_llm(self.take)["finetuned_prompt"], reviewed["finetuned_prompt"])

    def test_public_decision_is_none_when_only_llm_review_exists(self):
        media_review.save_llm(self.take, {"summary": "Only LLM", "finetuned_prompt": "Refined"})
        self.assertIsNone(media_review.public(self.take))
        self.assertEqual(media_review.public_llm(self.take)["summary"], "Only LLM")

    def test_score_take_reports_disabled_decision_flag(self):
        take_input = take_review.TakeReviewInput(take_path=self.take, prompt="A cat.")
        with patch.dict(os.environ, {"H3_DECISION": "0"}):
            with self.assertRaisesRegex(H3Error, "H3_DECISION=0"):
                take_review.score_take(take_input, api_key="gem-key")


class H3PageTakeReviewApiTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.addCleanup(setattr, JOBS, "store", JOBS.store)
        self.store = JOBS.store = WorkspaceStore(self.root / "state")
        self.runtime = RuntimeConfig(
            script_dir=REPO_ROOT,
            comfy_url="http://comfy",
            comfy_dir=self.root / "comfy",
            models_config=self.root / "models.json",
            outputs_dir=self.root / "out",
        )
        self.generated = []
        handlers = H3Handlers(H3Services(runtime_config=lambda: self.runtime, generate=self.generate, download_url=None))
        previous = JOBS.callbacks.get("h3")
        JOBS.callbacks["h3"] = owned_generation(
            handlers.generate_for_ui, "h3", ("batch_count", *GENERATION_FIELDS, "preset"), metadata_output=True)
        self.addCleanup(lambda: JOBS.callbacks.update(h3=previous) if previous else JOBS.callbacks.pop("h3", None))
        services = H3PageServices(
            root=lambda: self.root / "pages",
            resolve=handlers.resolve_request_settings,
            describe=handlers.describe_settings,
            enhance=lambda prompt, *args, **kwargs: (prompt, "OK"),
            runtime=lambda: self.runtime,
        )
        app = FastAPI()
        install_owner_cookie(app, self.store)
        install_h3_page(app, services, gpu=True)
        self.client = TestClient(app)
        self.client.get("/h3/api/boot")

    def generate(self, **values):
        self.generated.append(values)
        target = self.root / "out" / f"video-{len(self.generated)}.mp4"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"fake-video")
        yield None, "Sampling"
        yield str(target), "Done"

    def wait_run(self, kind: str):
        for _ in range(200):
            data = self.client.get("/h3/api/run?state=true").json()
            if data["runs"][kind] and not data["runs"][kind]["active"]:
                return data
            time.sleep(0.02)
        self.fail(f"{kind} run did not finish")

    def test_h3_review_and_score_use_same_refs_and_restore_sparse_tags_even_on_non_gemini_writer(self):
        # Switch writer to Local 8B to verify take review still runs Gemini-only.
        self.client.post("/h3/api/writer", json={"changes": {
            "prompt_writer_backend": "Local MiniMax-H3 8B",
            "gemini_prompt_model": "gemini-3.8-flash",
            "gemini_api_key": "gem-key",
            "review": False,
        }})
        self.client.post("/h3/api/settings", json={"action": "mode", "changes": {"mode": "Reference media"}})
        # Upload into sparse slot ref_image_2 -> card tag <Picture 2>, engine tag <Picture 1>.
        pic = self.root / "hero.png"
        Image.new("RGB", (64, 64), (20, 80, 120)).save(pic)
        with pic.open("rb") as handle:
            self.client.post("/h3/api/media/upload?slot=ref_image_2", files={"file": ("hero.png", handle)})
        self.client.post("/h3/api/prompt", json={"prompt": "<Picture 2> walks across the bridge."})
        self.assertEqual(self.client.post("/h3/api/generate").status_code, 200)
        done = self.wait_run("generate")
        video = done["state"]["results"]["videos"][0]
        self.assertEqual(video["ref_count"], 1)

        # Gemini sees dense <Picture 1> and returns <Picture 1>; H3 page must restore <Picture 2>.
        recorder = {}
        reply = _fake_gemini_review_payload(
            "subject_definitions:\n<Subject 1> is from <Picture 1>.\n\ndetailed_description:\n[Shot 1] <Subject 1> crosses slowly."
        )
        with patch.object(gemini, "Session", side_effect=lambda k: _FakeGeminiSession(k, reply_payload=reply, recorder=recorder)):
            resp = self.client.post("/h3/api/take/review", json={
                "path": video["path"],
                "advice": "Slow down the walking pace.",
                "model": "gemini-4-hydrogen",
            })
            self.assertEqual(resp.status_code, 200, resp.text)
            reviewed_state = self.wait_run("review")
        self.assertIsNone(reviewed_state["runs"]["review"]["error"])
        llm = reviewed_state["state"]["results"]["videos"][0]["llm_review"]
        self.assertEqual(llm["model"], "gemini-4-hydrogen")
        self.assertIn("<Picture 2>", llm["finetuned_prompt"])
        self.assertNotIn("<Picture 1>", llm["finetuned_prompt"])
        self.assertEqual(llm["ref_count"], 1)
        self.assertIn("<Picture 1> walks across the bridge.", recorder["generates"][0]["parts"][0]["text"])

        # Now score the exact same take with Gemini Decision + advice.
        with patch.dict(os.environ, {"H3_DECISION": "1"}), \
                patch.object(decisions, "decide", side_effect=_fake_decision_answers) as asked:
            resp = self.client.post("/h3/api/take/score", json={
                "path": video["path"],
                "advice": "Slow down the walking pace.",
            })
            self.assertEqual(resp.status_code, 200, resp.text)
            scored_state = self.wait_run("score")
        self.assertIsNone(scored_state["runs"]["score"]["error"])
        result_video = scored_state["state"]["results"]["videos"][0]
        self.assertEqual(result_video["review"]["score"], 80)
        self.assertEqual(result_video["review"]["scores"]["references"], 70)
        self.assertEqual(result_video["review"]["scores"]["advice"], 60)
        self.assertEqual(result_video["llm_review"]["model"], "gemini-4-hydrogen")
        media_blocks = [b for b in asked.call_args.args[1] if b.get("type") != "text"]
        self.assertEqual([b["type"] for b in media_blocks], ["image", "video"])

        # Security check: arbitrary files outside current results are refused.
        other = self.root / "secret.mp4"
        other.write_bytes(b"nope")
        refused = self.client.post("/h3/api/take/review", json={"path": str(other)})
        self.assertEqual(refused.status_code, 400)

    def test_h3_auto_background_score_includes_fl2va_frames_and_voice_refs(self):
        self.client.post("/h3/api/writer", json={"changes": {"gemini_api_key": "gem-key", "review": True}})
        self.client.post("/h3/api/settings", json={"action": "mode", "changes": {"mode": "First / last frame"}})
        pic = self.root / "first.png"
        Image.new("RGB", (64, 64), (10, 40, 80)).save(pic)
        with pic.open("rb") as handle:
            self.client.post("/h3/api/media/upload?slot=first_image", files={"file": ("first.png", handle)})
        voice = self.root / "voice.wav"
        voice.write_bytes(b"RIFFwav")
        with voice.open("rb") as handle:
            self.client.post("/h3/api/media/upload?slot=fl2va_audio_1", files={"file": ("voice.wav", handle)})
        self.client.post("/h3/api/prompt", json={"prompt": "She speaks with <Audio 1>."})

        seen_blocks = []

        def fake_decide(questions, blocks, *, key, **kwargs):
            seen_blocks.append(blocks)
            return _fake_decision_answers(questions, blocks, key=key, **kwargs)

        with patch.dict(os.environ, {"H3_DECISION": "1"}), \
                patch.object(decisions, "decide", side_effect=fake_decide):
            self.assertEqual(self.client.post("/h3/api/generate").status_code, 200)
            done = self.wait_run("generate")
            video_path = done["state"]["results"]["videos"][0]["path"]
            for _ in range(100):
                if (media_review.read(video_path) or {}).get("status") == media_review.DONE:
                    break
                time.sleep(0.02)

        self.assertEqual(len(seen_blocks), 1)
        media_types = [b["type"] for b in seen_blocks[0] if b.get("type") != "text"]
        self.assertEqual(media_types, ["image", "audio", "video"])
        shown = media_review.public(video_path)
        self.assertEqual(shown["score"], 80)
        self.assertEqual(shown["ref_count"], 2)


class StoryboardTakeReviewApiTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.addCleanup(setattr, JOBS, "store", JOBS.store)
        self.store = JOBS.store = WorkspaceStore(self.root / "state")
        self.runtime = RuntimeConfig(
            script_dir=REPO_ROOT,
            comfy_url="http://comfy",
            comfy_dir=self.root / "comfy",
            models_config=self.root / "models.json",
            outputs_dir=self.root / "outputs",
        )
        self.service = sb.StoryboardService(self.store, lambda: self.runtime)
        self.handlers = StoryboardHandlers(StoryboardServices(
            service=lambda: self.service,
            generate_take=self.fake_take,
            request_defaults=lambda: dict(PROFILE_VALUES),
            publish=lambda path, metadata: None,
            free_gpu=lambda: None,
        ))
        app = FastAPI()
        install_owner_cookie(app, self.store)
        install_storyboard_page(app, self.handlers, gpu=True)
        self.client = TestClient(app)

    def fake_take(self, values, graph, outputs):
        video = self.runtime.output_dir / f"{graph.output_prefix}_00001_.mp4"
        video.parent.mkdir(parents=True, exist_ok=True)
        video.write_bytes(b"mp4-video-audio")
        latent = f"{graph.latent_output_prefix}_00001_.safetensors"
        header = json.dumps({"__metadata__": {"h3_prefix_frames": "39"}}).encode()
        latent_path = self.runtime.output_dir / latent
        latent_path.parent.mkdir(parents=True, exist_ok=True)
        latent_path.write_bytes(struct.pack("<Q", len(header)) + header)
        outputs["latent"] = latent
        yield GenerationUpdate(str(video), "Completed")

    def wait_run(self, board_id: str):
        for _ in range(200):
            run = self.client.get(f"/storyboard/api/boards/{board_id}/run").json()["run"]
            if run and not run["active"]:
                return run
            time.sleep(0.01)
        self.fail("Storyboard run did not finish")

    def test_storyboard_take_review_and_score_include_cast_refs_and_strip_cast_tags(self):
        board = self.client.post("/storyboard/api/boards", json={
            "name": "Bridge", "story": "Mei steps onto the rainy bridge.",
        }).json()
        board_id = board["id"]
        scene_id = board["scenes"][0]["id"]

        # Add a cast member + switch scene to Reference media + attach audio reference.
        pic = self.root / "mei.png"
        Image.new("RGB", (128, 128), (120, 40, 80)).save(pic)
        with pic.open("rb") as handle:
            self.client.post(f"/storyboard/api/boards/{board_id}/cast", data={"name": "Mei"}, files={"file": ("mei.png", handle)})
        self.client.patch(f"/storyboard/api/boards/{board_id}/scenes/{scene_id}", json={"mode": "Reference media"})
        audio = self.root / "voice.wav"
        audio.write_bytes(b"RIFFwav")
        with audio.open("rb") as handle:
            self.client.post(f"/storyboard/api/boards/{board_id}/scenes/{scene_id}/media/ref_audios",
                             files=[("files", ("voice.wav", handle))])

        # Generate take 1 (auto-review disabled via no key, so we test explicit score + review).
        with patch.object(decisions, "key_or_none", return_value=None):
            self.assertEqual(self.client.post(
                f"/storyboard/api/boards/{board_id}/generate",
                json={"action": "scene", "scene_id": scene_id},
            ).status_code, 200)
            self.wait_run(board_id)

        updated = self.client.get(f"/storyboard/api/boards/{board_id}").json()
        take = updated["scenes"][0]["takes"][0]
        self.assertEqual(take["ref_count"], 2)

        # Gemini LLM review: sends cast picture + ref audio + take video, strips cast prefix from finetuned prompt.
        recorder = {}
        reply = _fake_gemini_review_payload(
            "Mei is <Picture 1>. Mei pauses by the railing and speaks softly using <Audio 1>."
        )
        with patch.object(gemini, "Session", side_effect=lambda k: _FakeGeminiSession(k, reply_payload=reply, recorder=recorder)):
            reviewed = self.client.post(
                f"/storyboard/api/boards/{board_id}/scenes/{scene_id}/takes/{take['id']}/review",
                json={"advice": "Have Mei pause before speaking.", "model": DEFAULT_GEMINI_PROMPT_MODEL, "gemini_api_key": "gem-key"},
            )
        self.assertEqual(reviewed.status_code, 200, reviewed.text)
        data = reviewed.json()
        llm = data["llm_review"]
        self.assertEqual(llm["ref_count"], 2)
        self.assertEqual(llm["finetuned_prompt"], "Mei pauses by the railing and speaks softly using <Audio 1>.")
        self.assertEqual(data["scenes"][0]["takes"][0]["llm_review"]["finetuned_prompt"], llm["finetuned_prompt"])
        # Verify Gemini received the cast tag in input prompt and 3 uploaded files (cast png, audio, take mp4).
        self.assertIn("Mei is <Picture 1>.", recorder["generates"][0]["parts"][0]["text"])
        self.assertEqual(len(recorder["uploads"]), 3)

        # Score the same take with Gemini Decision + advice.
        with patch.dict(os.environ, {"H3_DECISION": "1"}), \
                patch.object(decisions, "decide", side_effect=_fake_decision_answers) as asked:
            scored = self.client.post(
                f"/storyboard/api/boards/{board_id}/scenes/{scene_id}/takes/{take['id']}/score",
                json={"advice": "Have Mei pause before speaking.", "gemini_api_key": "gem-key"},
            )
        self.assertEqual(scored.status_code, 200, scored.text)
        scored_data = scored.json()
        self.assertEqual(scored_data["review"]["score"], 80)
        self.assertEqual(scored_data["review"]["scores"]["references"], 70)
        self.assertEqual(scored_data["review"]["scores"]["advice"], 60)
        # LLM review stays intact after scoring.
        self.assertEqual(scored_data["scenes"][0]["takes"][0]["llm_review"]["finetuned_prompt"], llm["finetuned_prompt"])
        media_blocks = [b for b in asked.call_args.args[1] if b.get("type") != "text"]
        self.assertEqual([b["type"] for b in media_blocks], ["image", "audio", "video"])

    def test_storyboard_take_review_and_score_work_in_cpu_workshop_mode(self):
        # Generate one take on GPU, then open the same store through a CPU-only Workshop app.
        board = self.client.post("/storyboard/api/boards", json={"name": "Night", "story": "Rain falls on neon streets."}).json()
        board_id, scene_id = board["id"], board["scenes"][0]["id"]
        with patch.object(decisions, "key_or_none", return_value=None):
            self.client.post(f"/storyboard/api/boards/{board_id}/generate", json={"action": "scene", "scene_id": scene_id})
            self.wait_run(board_id)
        take_id = self.client.get(f"/storyboard/api/boards/{board_id}").json()["scenes"][0]["takes"][0]["id"]

        cpu_app = FastAPI()
        install_owner_cookie(cpu_app, self.store)
        install_storyboard_page(cpu_app, self.handlers, gpu=False)
        cpu_client = TestClient(cpu_app, cookies=self.client.cookies)

        with patch.dict(os.environ, {"H3_DECISION": "1"}), \
                patch.object(gemini, "Session", side_effect=lambda k: _FakeGeminiSession(k)), \
                patch.object(decisions, "decide", side_effect=_fake_decision_answers):
            reviewed = cpu_client.post(
                f"/storyboard/api/boards/{board_id}/scenes/{scene_id}/takes/{take_id}/review",
                json={"gemini_api_key": "gem-key"},
            )
            scored = cpu_client.post(
                f"/storyboard/api/boards/{board_id}/scenes/{scene_id}/takes/{take_id}/score",
                json={"gemini_api_key": "gem-key"},
            )
        self.assertEqual(reviewed.status_code, 200, reviewed.text)
        self.assertEqual(scored.status_code, 200, scored.text)
        self.assertIn("finetuned_prompt", reviewed.json()["llm_review"])
        self.assertEqual(scored.json()["review"]["score"], 80)


if __name__ == "__main__":
    unittest.main()
