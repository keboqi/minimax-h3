"""Writer scripts are indented by staging depth without changing how they perform."""

import unittest

from h3_app.drama3_lint import lint_drama3_script
from h3_app.drama3_serving.tree import parse_drama3_tree
from h3_app.workflows.drama3 import format_script

SCRIPT = (
    'narrator: The door opened.<parallel><sound loop="true"  volume="0.2" name="rain">steady rain</sound>'
    "<sequential>Mia: Hello [softly] there.\n\nLeo: <phone_call>Who is it? <#1#> Answer me.</sequential>"
    "</parallel>narrator: Silence."
)


class FormatScriptTests(unittest.TestCase):
    def test_tags_and_turns_are_indented_by_depth(self):
        self.assertEqual(format_script(SCRIPT), "\n".join([
            "narrator: The door opened.",
            "<parallel>",
            '  <sound loop="true" volume="0.2" name="rain">steady rain</sound>',
            "  <sequential>",
            "    Mia: Hello [softly] there.",
            "    Leo: <phone_call>Who is it? <#1#> Answer me.",
            "  </sequential>",
            "</parallel>",
            "narrator: Silence.",
        ]))

    def test_formatting_keeps_the_parsed_performance(self):
        self.assertEqual(parse_drama3_tree(format_script(SCRIPT)), parse_drama3_tree(SCRIPT))

    def test_multiline_sound_bodies_join_onto_one_line(self):
        # The parser reads lines, so a sound split across lines only parses once joined.
        self.assertEqual(format_script('<sound name="rain">steady\nrain</sound>'),
                         '<sound name="rain">steady rain</sound>')

    def test_formatting_is_stable_and_leaves_plain_scripts_alone(self):
        formatted = format_script(SCRIPT)
        self.assertEqual(format_script(formatted), formatted)
        self.assertEqual(format_script("Mia: Hi.\nLeo: Hey."), "Mia: Hi.\nLeo: Hey.")

    def test_lint_repairs_indented_speaker_lines(self):
        script, _ = lint_drama3_script("<parallel>\n  <|speaker:0|> <whisper>Quiet</whisper> now.\n</parallel>")
        self.assertEqual(script.split("\n")[1][:2], "  ")
        self.assertNotIn("<|speaker:0|> <whisper>", script)


if __name__ == "__main__":
    unittest.main()
