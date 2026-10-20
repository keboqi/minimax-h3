"""Cast names must resolve unambiguously before any speech is generated."""

import unittest

from h3_app.errors import H3Error
from h3_app.generation.requests import Drama3Request
from h3_app.workflows.drama3 import names_to_tags, request_cast, speaker_label


class CastNameTests(unittest.TestCase):
    def test_duplicate_names_are_rejected_even_with_explicit_slot_lines(self):
        for second in ("Alex", " alex ", "ALEX"):
            with self.subTest(second=second):
                request = Drama3Request(
                    "Speaker 0: First voice.\nSpeaker 1: Second voice.",
                    voices=((0, "first.wav", "First."), (1, "second.wav", "Second.")),
                    voice_names=((0, "Alex"), (1, second)),
                )
                with self.assertRaisesRegex(H3Error, r"Speaker 0.*Speaker 1"):
                    request_cast(request)

    def test_writer_name_conversion_rejects_ambiguous_names(self):
        with self.assertRaises(H3Error):
            names_to_tags("Alex Smith: Hello.", {0: "Alex Smith", 2: " alex  smith "})

    def test_names_cannot_override_another_slots_generic_label(self):
        for name in ("Speaker 1", "speaker1", " SPEAKER 01 "):
            with self.subTest(name=name), self.assertRaisesRegex(H3Error, "reserved"):
                request_cast(Drama3Request("Speaker 1: Hello.", voice_names=((0, name),)))

    def test_distinct_and_unnamed_slots_keep_their_own_references(self):
        request = Drama3Request(
            "Alex: Hello.\nSpeaker 2: Reply.\nSpeaker 3: Goodbye.",
            voices=((0, "alex.wav", "Hi."), (2, "sam.wav", "Hey."), (3, "other.wav", "Bye.")),
            voice_names=((0, "Alex"), (2, "Sam"), (3, "")),
        )
        names, default, resolve = request_cast(request)
        references = {speaker_label(slot, names): path for slot, path, _ in request.voices}
        self.assertEqual(default, "Alex")
        for name, path in ((" alex ", "alex.wav"), ("Speaker 2", "sam.wav"), ("Speaker 3", "other.wav")):
            self.assertEqual(references[resolve(name)], path)
        self.assertEqual(
            names_to_tags(request.text, names),
            "<|speaker:0|> Hello.\n<|speaker:2|> Reply.\n<|speaker:3|> Goodbye.",
        )
        self.assertEqual(request_cast(Drama3Request("", voice_names=((1, "Speaker 1"),)))[2]("Speaker 1"),
                         "Speaker 1")


if __name__ == "__main__":
    unittest.main()
