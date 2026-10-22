"""Scoped voice reuse and the writer/bulk-save integration."""

import gradio as gr
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from h3_app.drama3 import speakers
from h3_app.errors import H3Error
from tests.test_drama3_voice_design import cast_entry


class SpeakerGroupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.runtime = SimpleNamespace(outputs_dir=self.root,
                                       prompt_systems={"Drama 3": Path("prompts/prompt_drama3.txt")})
        self.env = patch.dict(os.environ, {"H3_WORKSPACE_DIR": str(self.root)})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.clip = self.root / "voice.wav"
        self.clip.write_bytes(b"RIFF" + b"x" * 80)

    def save(self, name, group="Global"):
        return speakers.save_speaker(name, self.clip, "Exact spoken words.", self.runtime, "Warm voice", group)

    def test_groups_are_isolated_and_global_is_a_fallback(self):
        global_voice = self.save("Narrator")
        group_voice = self.save("narrator", "Story A")
        self.save("Alice", "Story B")
        self.assertEqual(speakers.create_group("story a", self.runtime), "Story A")
        self.assertEqual(speakers.list_groups(self.runtime), ["Global", "Story A", "Story B"])
        self.assertEqual(speakers.match_speaker("旁白", self.runtime, "Story A"), group_voice)
        self.assertEqual(speakers.match_speaker("story teller", self.runtime, "Story B"), global_voice)
        self.assertIsNone(speakers.match_speaker("Alice", self.runtime, "Story A"))
        self.assertEqual(speakers.load_speaker(group_voice.key, self.runtime), group_voice)
        speakers.delete_speaker(group_voice.key, self.runtime)
        self.assertEqual(speakers.match_speaker("narrator", self.runtime, "Story A"), global_voice)

    def test_matching_every_slot_scans_each_folder_once(self):
        from h3_ui.application import drama3_handlers

        self.save("Narrator")
        alice = self.save("Alice", "Story A")
        slots = []
        for name in ["Alice", "Narrator", *[""] * 14]:
            slots += [name, "Upload a clip", None]
        with patch.object(drama3_handlers, "_runtime_config", return_value=self.runtime), \
                patch.object(speakers, "list_speakers", wraps=speakers.list_speakers) as listed:
            updates = drama3_handlers.match_drama3_speakers("Story A", *slots)
        self.assertEqual(listed.call_count, 2)  # Story A, then Global
        self.assertEqual(updates[0]["value"], alice.key)
        self.assertEqual(updates[1]["value"], "Narrator")
        self.assertEqual(updates[2], gr.skip())

    def test_legacy_presets_and_path_like_group_names(self):
        saved = self.save("Alice")
        metadata = saved.audio.parent / "speaker.json"
        data = json.loads(metadata.read_text(encoding="utf-8"))
        del data["group"]
        metadata.write_text(json.dumps(data), encoding="utf-8")
        self.assertEqual(speakers.load_speaker("alice", self.runtime).group, "Global")
        scoped = self.save("Alice", "../../story")
        self.assertTrue(scoped.audio.is_relative_to(speakers.speakers_root(self.runtime)))
        self.assertEqual(speakers.load_speaker("Alice", self.runtime).audio, saved.audio)

    def test_rewrite_selects_saved_voices_and_keeps_new_cast(self):
        from h3_ui.application import drama3_handlers
        narrator = self.save("Narrator")
        alice = self.save("Alice", "Story")
        entries = [cast_entry(0, "旁白"), cast_entry(2, "ALICE"), cast_entry(5, "Bob")]
        reply = json.dumps({"script": "\n".join(f'<|speaker:{e["speaker_id"]}|> Hello there.' for e in entries),
                            "speakers": entries})
        slots = ["Upload a clip", None, ""] * 16 + [""] * 48 + [4, "Basic", "Story"]
        with patch.object(drama3_handlers, "_runtime_config", return_value=self.runtime):
            result = drama3_handlers.apply_drama3_writer_reply(reply, "Original", "Auto", *slots)
            self.assertEqual(len(result), 86)
            self.assertEqual(result[-17]["value"], narrator.key)
            self.assertEqual(result[-15]["value"], alice.key)
            self.assertEqual(result[-12], {"__type__": "update"})
            self.assertEqual(result[17], "Bob")
            self.assertEqual(result[50], 6)
            # A reply without a title keeps the title box as it is.
            self.assertEqual(result[-1], {"__type__": "update"})
            titled = drama3_handlers.apply_drama3_writer_reply(
                json.dumps({"title": "第一章 <重逢>\n", **json.loads(reply)}), "Original", "Auto", *slots)
            self.assertEqual(titled[-1], "第一章 重逢")
            bad = drama3_handlers.apply_drama3_writer_reply("invalid", "Original", "Auto", *slots)
            self.assertEqual(len(bad), len(result))
            self.assertTrue(all(update == {"__type__": "update"} for update in bad[2:]))
            context = drama3_handlers._drama3_writer_slots(slots)[2]
            self.assertIn("Reusable speaker voices", context)
            self.assertIn("Alice", context)

    def test_bulk_save_copies_presets_and_reports_invalid_clips(self):
        from h3_ui.application import drama3_handlers
        saved = self.save("Narrator")
        slots = [saved.key, "Narrator", None, "", "",
                 "Upload a clip", "Alice", str(self.clip), "Alice's words.", "Bright voice",
                 "Upload a clip", "Bob", str(self.clip), "", ""]
        with patch.object(drama3_handlers, "_runtime_config", return_value=self.runtime):
            result = drama3_handlers.save_all_drama3_speakers("Story", 3, *slots)
        self.assertIn("Saved 2 speakers", result[0])
        self.assertIn("Speaker 2:", result[0])
        self.assertEqual(speakers.load_speaker(result[1]["value"], self.runtime).group, "Story")
        self.assertEqual(speakers.load_speaker(result[2]["value"], self.runtime).transcript, "Alice's words.")
        self.assertEqual(speakers.load_speaker(saved.key, self.runtime), saved)

    def test_matching_preserves_uploaded_references(self):
        from h3_ui.application import drama3_handlers
        saved = self.save("Alice", "Story")
        with patch.object(drama3_handlers, "_runtime_config", return_value=self.runtime):
            self.assertEqual(drama3_handlers.match_drama3_speaker("Story", "alice", "Upload a clip", None)["value"], saved.key)
            self.assertEqual(drama3_handlers.match_drama3_speaker("Story", "alice", "Upload a clip", str(self.clip)),
                             {"__type__": "update"})

    def test_canonical_narrator_does_not_reassign_an_existing_alias(self):
        from h3_ui.application import drama3_handlers
        with patch.object(drama3_handlers, "_runtime_config", return_value=self.runtime):
            result = drama3_handlers._drama3_apply_cast(
                "<|speaker:0|> Hello.", "Rewritten", [cast_entry(0, "narrator")],
                {0: {"name": "旁白", "has_reference": True}}, 4, "Global")
        self.assertEqual(result[2], "旁白")
        self.assertEqual(result[-16], {"__type__": "update"})

    def test_library_rename_copy_and_collision_protection(self):
        original = self.save("Alice", "Story")
        other = self.save("Bob", "Story")
        with self.assertRaises(H3Error):
            speakers.update_speaker(original.key, "Bob", "Words.", "Voice", self.runtime)
        updated = speakers.update_speaker(original.key, "Alicia", "Correct words.", "Gentle", self.runtime)
        self.assertEqual(updated.audio.read_bytes(), self.clip.read_bytes())
        self.assertEqual((updated.description, updated.transcript), ("Gentle", "Correct words."))
        with self.assertRaises(H3Error):
            speakers.load_speaker(original.key, self.runtime)
        copied = speakers.copy_speaker(updated.key, "Other story", self.runtime)
        self.assertEqual(copied.audio.read_bytes(), updated.audio.read_bytes())
        with self.assertRaises(H3Error):
            speakers.copy_speaker(updated.key, "Other story", self.runtime)
        self.assertEqual(speakers.load_speaker(other.key, self.runtime), other)

    def test_library_updates_selected_slots_after_rename_and_delete(self):
        from h3_ui.drama3.speaker_library import SpeakerLibraryActions
        actions = SpeakerLibraryActions(lambda: self.runtime)
        original = self.save("Alice", "Story")
        preview = actions.select(original.key)
        self.assertEqual(preview[0], str(original.audio))
        result = actions.mutate("save", "Story", original.key, "Alicia", "Gentle", "Exact words.",
                                "Global", original.key, "Upload a clip")
        renamed_key = result[0]["value"]
        self.assertEqual(result[2]["value"], renamed_key)
        self.assertEqual(result[3]["value"], "Upload a clip")
        deleted = actions.mutate("delete", "Story", renamed_key, "", "", "", "Global", renamed_key)
        self.assertIsNone(deleted[0]["value"])
        self.assertEqual(deleted[2]["value"], "Upload a clip")
        self.assertEqual(actions.choices("Story"), [])

    def test_library_toggles_and_keeps_saved_voice_until_a_try_is_chosen(self):
        from h3_ui.drama3.speaker_library import HIDE_LABEL, KEEP_SAVED, MANAGE_LABEL, SpeakerLibraryActions
        actions = SpeakerLibraryActions(lambda: self.runtime)
        alice = self.save("Alice", "Story")
        opened = actions.toggle(False, "Story")
        self.assertEqual((opened[0], opened[1]["visible"], opened[2]["value"]), (True, True, HIDE_LABEL))
        closed = actions.toggle(True, "Story")
        self.assertEqual((closed[0], closed[1]["visible"], closed[2]["value"]), (False, False, MANAGE_LABEL))
        take = self.root / "take.wav"
        take.write_bytes(b"RIFF" + b"y" * 80)
        brief = {"key": alice.key, "description": "Brighter", "transcript": "New words."}
        kept = actions.use_take(alice.key, KEEP_SAVED, brief, str(take))
        self.assertEqual(kept[-1], "Kept the saved voice.")
        self.assertEqual(speakers.load_speaker(alice.key, self.runtime).audio.read_bytes(), self.clip.read_bytes())
        stale = actions.use_take(alice.key, "New voice 1", {**brief, "key": "Story/Bob"}, str(take))
        self.assertIn("another speaker", stale[-1])
        used = actions.use_take(alice.key, "New voice 1", brief, str(take), None)
        saved = speakers.load_speaker(alice.key, self.runtime)
        self.assertEqual(saved.audio.read_bytes(), take.read_bytes())
        self.assertEqual((saved.description, saved.transcript), ("Brighter", "New words."))
        self.assertEqual(used[:3], (str(saved.audio), "Brighter", "New words."))
        self.assertEqual(used[3]["value"], KEEP_SAVED)

    def test_library_list_follows_saves_and_keeps_its_selection(self):
        from h3_ui.drama3.speaker_library import SpeakerLibraryActions
        actions = SpeakerLibraryActions(lambda: self.runtime)
        alice = self.save("Alice", "Story")
        listed, groups = actions.refresh_saved("Story", alice.key)
        self.assertEqual((listed["choices"], listed["value"]), ([("Alice", alice.key)], alice.key))
        self.assertEqual(groups["choices"], ["Global", "Story"])
        bob = self.save("Bob", "Story")
        listed, _groups = actions.refresh_saved("Story", alice.key)
        self.assertEqual(listed["choices"], [("Alice", alice.key), ("Bob", bob.key)])
        self.assertEqual(listed["value"], alice.key)
        speakers.delete_speaker(alice.key, self.runtime)
        self.assertIsNone(actions.refresh_saved("Story", alice.key)[0]["value"])

    def test_group_deletion_removes_owned_voices_and_resets_selections(self):
        from h3_ui.drama3.speaker_library import SpeakerLibraryActions
        actions = SpeakerLibraryActions(lambda: self.runtime)
        narrator = self.save("narrator")
        alice = self.save("Alice", "Story")
        bob = self.save("Bob", "Story")
        copied = speakers.copy_speaker(alice.key, "Other story", self.runtime)
        result = actions.delete_group("Story", alice.key, bob.key, narrator.key, copied.key, "Upload a clip")
        self.assertEqual(result[0]["value"], "Global")
        self.assertIsNone(result[1]["value"])
        self.assertEqual(result[2]["value"], "Global")
        self.assertIn("2 saved speakers", result[3])
        self.assertEqual([update["value"] for update in result[4:]],
                         ["Upload a clip", "Upload a clip", narrator.key, copied.key, "Upload a clip"])
        self.assertFalse(speakers.group_root(self.runtime, "Story").exists())
        self.assertFalse(alice.audio.exists())
        self.assertFalse(bob.audio.exists())
        self.assertEqual(speakers.list_groups(self.runtime), ["Global", "Other story"])
        self.assertEqual(speakers.load_speaker(copied.key, self.runtime), copied)
        self.assertEqual(speakers.load_speaker(narrator.key, self.runtime), narrator)

    def test_empty_group_deletion_and_global_protection(self):
        speakers.create_group("Empty", self.runtime)
        self.assertEqual(speakers.delete_group("Empty", self.runtime), [])
        saved = self.save("narrator")
        with self.assertRaises(H3Error):
            speakers.delete_group("Global", self.runtime)
        with self.assertRaises(H3Error):
            speakers.delete_group("Missing", self.runtime)
        self.assertTrue(saved.audio.is_file())


if __name__ == "__main__":
    unittest.main()
