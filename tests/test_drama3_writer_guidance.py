"""Writer guidance from Fish Audio's Drama 3 docs: reference-quality previews, briefed cues, spoken
parentheses, control scopes, and the documented sound events the app treats as sounds."""

import json
from pathlib import Path
import re
import unittest

from h3_app.drama3.lint import lint_drama3_script
from h3_app.drama3.serving.planner import AUDIBLE_EVENTS, delivery_cues

PROMPTS = Path(__file__).resolve().parents[1] / "prompts"
EVENTS = ("[small laugh]", "[chuckles]", "[shaky breath]", "[coughs]")


class WriterGuidanceTests(unittest.TestCase):
    def setUp(self):
        self.writer = (PROMPTS / "prompt_drama3.txt").read_text(encoding="utf-8")
        self.director = (PROMPTS / "prompt_drama3_direction.txt").read_text(encoding="utf-8")

    def test_previews_are_calm_reference_recordings_of_ten_to_twenty_seconds(self):
        self.assertIn("Preview speech becomes the voice's reference recording", self.writer)
        self.assertIn("about 10–20 seconds aloud", self.writer)
        self.assertIn("Never use the character's most emotional moment", self.writer)
        # The contract's own example preview follows the advice: several sentences, not one.
        contract = json.loads(re.search(r"^Output contract:\n(\{.*\})$", self.writer, re.M).group(1))
        preview = contract["speakers"][0]["preview_text"]
        self.assertGreaterEqual(len(re.findall(r"[.!?]", preview)), 3)
        self.assertGreaterEqual(len(preview.split()), 30)

    def test_cues_brief_intent_and_addressee_in_both_prompts(self):
        for prompt in (self.writer, self.director):
            self.assertIn("[to the child, gently]", prompt)
            self.assertIn("A bare emotion word is the weakest cue", prompt)

    def test_parentheses_are_spoken_and_scopes_are_explained(self):
        self.assertIn("includes text in round parentheses", self.writer)
        self.assertIn("（笑）", self.writer)
        self.assertIn("Fish also documents a style prefix", self.writer)
        self.assertIn("this script never opens a line with a cue", self.writer)

    def test_documented_sound_events_are_listed_and_treated_as_sounds(self):
        for event in EVENTS:
            with self.subTest(event=event):
                self.assertIn(event, self.writer)
                self.assertIn(event, self.director)
                # A sound event may close a line; the lint keeps it there.
                script, _notes = lint_drama3_script(f"<|speaker:0|> I'm fine, [softly] really. {event}")
                self.assertTrue(script.endswith(event), script)
                # It is never carried into the next request as the turn's delivery.
                self.assertEqual(delivery_cues(f"Well, {event} fine."), [])
        script, _notes = lint_drama3_script("<|speaker:0|> Get back to class! [angry]")
        self.assertNotIn("[angry]", script)
        # A plain [breath] stays a pause cue (Drama 3 may hold it as silence), so it is not offered.
        self.assertNotIn("[breath]", self.writer)
        self.assertNotIn("[breath]", lint_drama3_script("<|speaker:0|> Fine, really. [breath]")[0])

    def test_offered_sound_events_are_exactly_the_ones_the_app_treats_as_sounds(self):
        # A delivery cue offered as a sound event ends lines, where the lint strips it.
        for prompt in (self.writer, self.director):
            listed = re.search(r"^- Audible events[^:]*: (.*)$", prompt, re.M).group(1)
            offered = {cue.lower() for cue in re.findall(r"\[([^\]]+)\]", listed)}
            self.assertEqual(offered, set(AUDIBLE_EVENTS))
            delivery = re.search(r"^- Delivery and emotion[^:]*: (.*)$", prompt, re.M).group(1)
            self.assertIn("[whispering]", delivery)

    def test_preserved_lines_are_never_padded_with_invented_reactions(self):
        self.assertIn("when preserving supplied lines, never invent words", self.writer)
        self.assertIn("The one exception is an expansion the user explicitly asks for", self.writer)
        for prompt in (self.writer, self.director):
            self.assertNotIn("Never give a line the same cue as the line before", prompt)
            self.assertIn("Never copy the previous line's cue word for word", prompt)

    def test_staging_excerpt_gives_sounds_a_job_and_lints_clean(self):
        from h3_app.drama3.serving.tree import iter_sound_nodes, parse_drama3_tree

        self.assertIn("Give every sound a job", self.writer)
        example = self.writer.split("Advanced staging excerpt", 1)[1]
        script = example.split("Script:\n", 1)[1].split("\nWhy:", 1)[0].strip()
        self.assertEqual(lint_drama3_script(script), (script, []))
        tree = parse_drama3_tree(script)
        self.assertEqual({sound.name for sound in iter_sound_nodes(tree)},
                         {"coastal_storm", "approaching_boots", "radio_static"})


if __name__ == "__main__":
    unittest.main()
