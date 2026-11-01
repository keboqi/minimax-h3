"""A speaker's continuous passage stays one take unless staging cuts it; the writer is told so."""

from pathlib import Path
import unittest

from h3_app.drama3.serving.planner import plan_turns_chunks
from h3_app.drama3.serving.tree import iter_speech_nodes, parse_drama3_tree

PASSAGE = ["雨下了一整夜。", "陈念推开门。", "屋里一片漆黑。"]
SOUND = '<sound name="door_creak" duration="3">Slow creaking wooden door, close</sound>'


def requests(script: str) -> list[str]:
    """The Drama 3 requests a script becomes, as the generator plans them."""
    tree = parse_drama3_tree(script, default_speaker="Speaker 0")
    return [chunk.text for node in iter_speech_nodes(tree) for chunk in plan_turns_chunks(node.turns)]


class PassageTests(unittest.TestCase):
    def test_a_passage_in_one_block_is_one_request(self):
        lines = "\n".join(f"<|speaker:0|> {line}" for line in PASSAGE)
        self.assertEqual(requests(lines), ["".join(PASSAGE)])
        # Sound at the start: the whole passage sits in the sound's block and stays whole.
        whole = f"<parallel>\n{SOUND}\n<sequential>\n{lines}\n</sequential>\n</parallel>"
        self.assertEqual(requests(whole), ["".join(PASSAGE)])
        # Sound after the passage: a standalone sound line leaves it whole too.
        self.assertEqual(requests(f"{lines}\n{SOUND}"), ["".join(PASSAGE)])

    def test_a_mid_passage_block_cuts_the_passage_into_cold_starts(self):
        first, middle, last = (f"<|speaker:0|> {line}" for line in PASSAGE)
        pinned = f"{first}\n<parallel>\n{SOUND}\n<sequential>\n{middle}\n</sequential>\n</parallel>\n{last}"
        self.assertEqual(requests(pinned), PASSAGE)
        self.assertEqual(requests(f"{first}\n<#0.8#>\n{middle}\n{last}"), [PASSAGE[0], "".join(PASSAGE[1:])])

    def test_writer_prompt_keeps_passages_whole_and_times_events_inside_them(self):
        prompt = (Path(__file__).resolve().parents[1] / "prompts" / "prompt_drama3.txt").read_text(encoding="utf-8")
        self.assertIn("Keep consecutive lines of one speaker together in the speech timeline", prompt)
        # Events and holds inside a passage use phrase anchors on the parallel, not split blocks.
        self.assertIn('put the sound directly on the enclosing parallel with `anchor="exact spoken phrase"`', prompt)
        self.assertIn("For a hold inside one continuous speaker passage, use `holds=", prompt)


if __name__ == "__main__":
    unittest.main()
