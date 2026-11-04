"""Workshop/Studio handoff keeps drafts, edited lines, settings and completed results."""
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import gradio as gr

from h3_app.dubbing import pipeline, projects, session as sessions
from h3_app.dubbing.results import Player, player_folder, save_player
from h3_app.errors import H3Error
from h3_ui.dubbing.handlers import DubbingHandlers, DubbingServices
from h3_ui.dubbing.projects import bind_projects
from h3_ui.dubbing.view import build_dubbing_view
from h3_ui.dubbing.workshop import build_dubbing_workshop


class DubbingProjectTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.runtime = SimpleNamespace(outputs_dir=self.root / "outputs", output_dir=self.root / "outputs")
        env = patch.dict(os.environ, {"H3_WORKSPACE_DIR": str(self.root / "workspace")})
        env.start()
        self.addCleanup(env.stop)
        self.source = self.root / "interview.wav"
        self.source.write_bytes(b"uploaded media")
        self.actions = DubbingHandlers(DubbingServices(lambda: self.runtime, None, lambda mode: [], gpu=False))

    def test_saved_draft_outlives_upload_and_session_cleanup(self):
        draft = projects.save(self.runtime, "Interview", "", str(self.source), {"notes": "Keep names"})
        self.source.unlink()
        sessions.prune(self.runtime, now=10**12)
        data, loaded = projects.load(self.runtime, "Interview")
        self.assertEqual(loaded.id, draft.id)
        self.assertEqual(loaded.path("source").read_bytes(), b"uploaded media")
        self.assertEqual(data["settings"]["notes"], "Keep names")
        self.assertEqual(projects.names(self.runtime), ["Interview"])

    def test_analysis_handoff_and_line_edits_keep_takes_and_result(self):
        projects.save(self.runtime, "Interview", "", self.source, {})
        analyzed = sessions.create(self.runtime, self.source, video=False)
        analyzed.segments = [{"start": 0, "end": 1, "source": "Hi", "translation": "Bonjour"}]
        analyzed.data["last_output"] = "finished.mp3"
        analyzed.save()
        (analyzed.root / "takes").mkdir()
        take = analyzed.root / "takes" / "cached.wav"
        take.write_bytes(b"generated take")
        projects.attach(self.runtime, "Interview", analyzed.id)
        self.actions.save_lines(analyzed.id, [[1, "0", "1", "speaker1", "Hi", "Salut", True]])
        _, loaded = projects.load(self.runtime, "Interview")
        self.assertEqual(loaded.segments[0]["translation"], "Salut")
        self.assertEqual(loaded.data["last_output"], "finished.mp3")
        self.assertEqual(take.read_bytes(), b"generated take")
        saved = projects.save(self.runtime, "Interview", loaded.id, loaded.path("source"), {})
        self.assertEqual(saved.id, loaded.id)
        cached = self.root / "gradio-cache.wav"
        cached.write_bytes(self.source.read_bytes())
        saved = projects.save(self.runtime, "Interview", loaded.id, cached, {})
        self.assertEqual(saved.id, loaded.id)
        self.assertEqual(saved.segments[0]["translation"], "Salut")
        cached.write_bytes(b"different source")
        replaced = projects.save(self.runtime, "Interview", loaded.id, cached, {})
        self.assertNotEqual(replaced.id, loaded.id)
        self.assertEqual(replaced.segments, [])

    def test_workshop_and_studio_share_source_and_settings(self):
        with gr.Blocks() as cpu:
            view = build_dubbing_workshop(gr.Tab("Dub & Translate"), cpu, self.runtime)
        gpu_controls = (view.analyze, view.run, view.preview, view.new_take, view.retranslate, view.align_words)
        self.assertTrue(all(not control.visible for control in gpu_controls))
        self.assertFalse(any(fn.concurrency_id == "h3-gpu" for fn in cpu.fns.values()))
        targets = {target[0] for fn in cpu.fns.values() for target in fn.targets}
        self.assertTrue(all(control._id not in targets for control in gpu_controls))
        save = next(fn for fn in cpu.fns.values() if (view.project.save._id, "click") in fn.targets)
        values = [control.value for control in save.inputs]
        for control, value in ((view.project.name, "Long interview"), (view.source, str(self.source)),
                               (view.target_language, "French"), (view.notes, "Keep names"),
                               (view.max_speed, 1.3), (view.timing, "Natural pace")):
            values[save.inputs.index(control)] = value
        save.fn(*values)
        self.source.unlink()
        with gr.Blocks() as studio:
            restored = build_dubbing_view(gr.Column(), gemini_models=["model"], default_gemini_model="model")
            bind_projects(restored, self.actions, studio.load)
        load = next(fn for fn in studio.fns.values() if (restored.project.load._id, "click") in fn.targets)
        loaded = dict(zip(load.outputs, load.fn("Long interview"), strict=True))
        self.assertEqual(loaded[restored.target_language], "French")
        self.assertEqual(loaded[restored.notes], "Keep names")
        self.assertEqual(loaded[restored.max_speed], 1.3)
        self.assertEqual(loaded[restored.timing], "Natural pace")
        self.assertEqual(loaded[restored.api_key], "")
        self.assertEqual(Path(loaded[restored.source]).read_bytes(), b"uploaded media")

    def test_workshop_rejects_model_execution(self):
        run = Mock()
        with self.assertRaisesRegex(H3Error, "Studio"):
            self.actions._gpu(run)
        run.assert_not_called()

    def test_replaced_project_sessions_expire_but_current_session_survives(self):
        draft = projects.save(self.runtime, "Interview", "", self.source, {})
        analyzed = sessions.create(self.runtime, self.source, video=False)
        projects.attach(self.runtime, "Interview", analyzed.id, expected_session=draft.id)
        # Also migrate sessions retained by the earlier boolean flag implementation.
        sessions.modify(draft, lambda current: current.data.update(project=True))
        sessions.prune(self.runtime, now=10**12)
        self.assertFalse(draft.root.exists())
        self.assertTrue(analyzed.root.exists())

    def test_analysis_attaches_to_captured_project_without_browser_change_callback(self):
        draft = projects.save(self.runtime, "Interview", "", self.source, {})
        analyzed = sessions.create(self.runtime, draft.path("source"), video=False)
        selected = {"name": "Interview", "session": draft.id}
        with patch.object(self.actions, "_gpu", return_value=iter([pipeline.Analysis(analyzed, "Done")])):
            updates = list(self.actions.analyze_source(
                str(draft.path("source")), None, True, "Balanced (BS-Roformer)", pipeline.PIPELINE_LOCAL,
                "Auto-detect", "French", True, pipeline.TRANSLATOR_HYMT, True, .35, "model", "", "",
                project=selected))
        self.assertEqual(updates[-1][0], analyzed.id)
        self.assertEqual(projects.load(self.runtime, "Interview")[1].id, analyzed.id)
        self.assertEqual(projects.load(self.runtime, "Interview")[1].data["name"], "interview")
        self.assertEqual(self.actions.project_selection(selected, analyzed.id),
                         {"name": "Interview", "session": analyzed.id})

    def test_opening_unrelated_session_never_repoints_project(self):
        draft = projects.save(self.runtime, "Interview", "", self.source, {})
        other = sessions.create(self.runtime, self.source, video=False)
        selected = {"name": "Interview", "session": draft.id}
        self.assertIsNone(self.actions.project_selection(selected, other.id))
        self.assertEqual(projects.load(self.runtime, "Interview")[1].id, draft.id)

    def test_project_write_failure_still_returns_completed_analysis(self):
        draft = projects.save(self.runtime, "Interview", "", self.source, {})
        analyzed = sessions.create(self.runtime, self.source, video=False)
        with (patch.object(self.actions, "_gpu", return_value=iter([pipeline.Analysis(analyzed, "Done")])),
              patch.object(projects, "attach", side_effect=OSError("volume unavailable"))):
            updates = list(self.actions.analyze_source(
                str(self.source), None, True, "Balanced (BS-Roformer)", pipeline.PIPELINE_LOCAL,
                "Auto-detect", "French", True, pipeline.TRANSLATOR_HYMT, True, .35, "model", "", "",
                project={"name": "Interview", "session": draft.id}))
        self.assertEqual(updates[-1][0], analyzed.id)
        self.assertIn("Project could not be updated", updates[-1][3])

    def test_stale_analysis_cannot_overwrite_newer_project_session(self):
        draft = projects.save(self.runtime, "Interview", "", self.source, {})
        newer = sessions.create(self.runtime, self.source, video=False)
        projects.attach(self.runtime, "Interview", newer.id, expected_session=draft.id)
        stale = sessions.create(self.runtime, self.source, video=False)
        self.assertFalse(projects.attach(self.runtime, "Interview", stale.id, expected_session=draft.id))
        self.assertEqual(projects.load(self.runtime, "Interview")[1].id, newer.id)

    def test_session_reset_preserves_loaded_generation_settings_only_for_that_session(self):
        draft = projects.save(self.runtime, "Interview", "", self.source, {})
        settings = {"session": draft.id, "regenerate": "2, 4", "voice_mode": pipeline.VOICE_NEAREST}
        self.assertEqual(self.actions.clear_results(draft.id, settings)[3], "2, 4")
        self.assertEqual(self.actions.voice_default(draft.id, settings), pipeline.VOICE_NEAREST)
        other = sessions.create(self.runtime, self.source, video=False)
        self.assertEqual(self.actions.clear_results(other.id, settings)[3], "")
        self.assertEqual(self.actions.voice_default(other.id, settings), pipeline.default_voice_mode(None))

    def test_reopen_completed_result_without_models(self):
        session = projects.save(self.runtime, "Interview", "", self.source, {})
        folder = self.runtime.outputs_dir / "dubbing"
        folder.mkdir(parents=True)
        output = folder / "interview.mp3"
        output.write_bytes(b"result")
        subtitle = folder / "interview.french.srt"
        subtitle.write_text("1\n00:00:00,000 --> 00:00:01,000\nSalut\n", encoding="utf-8")
        (folder / "interview2.french.srt").write_text("Another dub", encoding="utf-8")
        tracks = player_folder(output)
        tracks.mkdir(parents=True)
        dub, original = tracks / "dub.mp3", tracks / "original.mp3"
        dub.write_bytes(b"dub")
        original.write_bytes(b"original")
        save_player(tracks, Player(None, dub, original, None, 1, []))
        sessions.modify(session, lambda current: current.data.update(last_output=str(output)))
        values = self.actions.restore_result(session.id)
        self.assertTrue(values[0]["visible"])
        self.assertEqual(values[2], str(output))
        self.assertEqual(values[3], [str(subtitle)])
        sessions.modify(session, lambda current: current.data.update(last_output=str(output.with_suffix(".mp4"))))
        self.assertEqual(self.actions.restore_result(session.id)[2], str(output))


if __name__ == "__main__":
    unittest.main()
