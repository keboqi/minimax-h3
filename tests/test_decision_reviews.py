"""Gemini Decision checks: the client, source-change labels, dub translation checks, combine
suggestions, and prompt-fit reviews with library tags. No request leaves the machine."""

from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import time
import unittest
from unittest.mock import MagicMock, patch

from h3_app.dubbing import review_checks
from h3_app.errors import H3Error
from h3_app.llm import decisions

from tests.test_dub_page import API, PageTests


def response(status=200, payload=None):
    reply = MagicMock()
    reply.ok = status < 400
    reply.status_code = status
    reply.json.return_value = payload if payload is not None else {}
    reply.text = json.dumps(payload or {})
    return reply


class DecisionClientTests(unittest.TestCase):
    def test_questions_are_chunked_and_answers_merged(self):
        http = MagicMock()
        sent = []

        def post(url, headers, json, timeout):
            sent.append(json)
            return response(payload={"answers": {name: {"type": "noul", "noul": 0.9, "confidence": 0.9}
                                                 for name in json["questions"]}})

        http.post.side_effect = post
        questions = {f"q{i}": decisions.yes_no(f"Question {i}?") for i in range(5)}
        answers = decisions.decide(questions, "shared input", key="k", chunk=2, http=http)
        self.assertEqual(len(sent), 3)
        self.assertEqual(sent[0]["model"], decisions.MODEL)
        self.assertEqual(sent[0]["input"], "shared input")
        self.assertEqual(set(answers), set(questions))
        self.assertEqual(decisions.yes_probability(answers["q3"]), 0.9)

    def test_a_refused_chunk_is_split_and_the_blocked_question_skipped(self):
        http = MagicMock()

        def post(url, headers, json, timeout):
            if "bad" in json["questions"]:
                return response(400, {"error": {"code": "safety", "message": "blocked"}})
            return response(payload={"answers": {name: {"type": "noul", "noul": 0.2} for name in json["questions"]}})

        http.post.side_effect = post
        questions = {"a": decisions.yes_no("A?"), "bad": decisions.yes_no("B?"), "c": decisions.yes_no("C?")}
        self.assertEqual(set(decisions.decide(questions, "x", key="k", http=http)), {"a", "c"})
        # The blocked question first: the others are still answered.
        first = {"bad": decisions.yes_no("B?"), "a": decisions.yes_no("A?"), "c": decisions.yes_no("C?")}
        self.assertEqual(set(decisions.decide(first, "x", key="k", http=http)), {"a", "c"})

    def test_a_dropped_connection_is_retried(self):
        import requests

        http = MagicMock()
        http.post.side_effect = [requests.ConnectionError("reset"),
                                 response(payload={"answers": {"q": {"type": "noul", "noul": 0.7}}})]
        with patch.object(decisions.time, "sleep"):
            answers = decisions.decide({"q": decisions.yes_no("?")}, "x", key="k", http=http)
        self.assertEqual(decisions.yes_probability(answers["q"]), 0.7)

    def test_a_blocked_input_is_not_retried_for_every_question(self):
        http = MagicMock()
        http.post.return_value = response(400, {"error": {"code": "image_safety", "message": "blocked"}})
        with self.assertRaisesRegex(H3Error, "HTTP 400"):
            decisions.decide({f"q{i}": decisions.yes_no("?") for i in range(200)}, "x", key="k", http=http)
        self.assertLessEqual(http.post.call_count, 10)

    def test_a_bad_key_stops_at_once(self):
        http = MagicMock()
        http.post.return_value = response(403, {"error": {"message": "API key not valid"}})
        with self.assertRaisesRegex(H3Error, "HTTP 403"):
            decisions.decide({f"q{i}": decisions.yes_no("?") for i in range(8)}, "x", key="k", http=http)
        self.assertEqual(http.post.call_count, 1)

    def test_score_fraction_and_key_switch(self):
        answer = {"type": "score", "score": 3.0, "legend": {str(i): "" for i in range(5)}}
        self.assertEqual(decisions.score_fraction(answer), 0.75)
        self.assertIsNone(decisions.score_fraction({"type": "noul", "noul": 0.5}))
        with patch.dict(os.environ, {"H3_DECISION": "1", "GEMINI_API_KEY": "env"}):
            self.assertEqual(decisions.key_or_none(""), "env")
            self.assertEqual(decisions.key_or_none("typed"), "typed")
        with patch.dict(os.environ, {"H3_DECISION": "0", "GEMINI_API_KEY": "env"}):
            self.assertIsNone(decisions.key_or_none("typed"))


class SourceChangeLabelTests(unittest.TestCase):
    SOURCE = "She walked to the old mill and opened the heavy door."
    SCRIPT = "Speaker 1: She walked to the old mill and closed the heavy door."

    def test_meaning_changes_are_named_before_typo_fixes(self):
        from h3_app.drama3.guided import fidelity

        changes = [("words", "opened", "closed", 0.05), ("words", "teh", "the", 0.97)]
        note = fidelity.source_change_note(changes)
        self.assertIn("1 may change the meaning (opened → closed)", note)
        self.assertIn("1 looks like typo fixes (teh → the)", note)
        self.assertIn("Review them", fidelity.source_change_note([("words", "a", "b")]))

    def test_labels_come_from_one_decision_request(self):
        from h3_app.drama3.guided import fidelity

        with patch.object(fidelity, "_decision_key", return_value="k"), \
                patch.object(decisions, "decide", return_value={"change_0": {"type": "noul", "noul": 0.1}}) as asked:
            changes = fidelity.source_changes(self.SOURCE, self.SCRIPT, label=True)
        self.assertEqual(asked.call_count, 1)
        self.assertEqual(changes[0][0], "words")
        self.assertAlmostEqual(changes[0][3], 0.1)

    def test_strict_check_accepts_only_confident_repairs(self):
        from h3_app.drama3.guided import fidelity

        source, script = "I will recieve it.", "Speaker 1: I will receeve it."
        for probability, accepted in ((0.95, True), (0.4, False)):
            with self.subTest(probability=probability), patch.object(fidelity, "_decision_key", return_value="k"), \
                    patch.object(decisions, "decide", return_value={"change_0": {"type": "noul", "noul": probability}}):
                if accepted:
                    self.assertTrue(fidelity.validate_source_fidelity(source, script))
                else:
                    with self.assertRaises(H3Error):
                        fidelity.validate_source_fidelity(source, script)


class CombineSuggestionTests(unittest.TestCase):
    def line(self, start, end, source, speaker="speaker1", **extra):
        return {"start": start, "end": end, "speaker": speaker, "source": source, "translation": "", **extra}

    def test_split_sentences_and_fragments_of_one_speaker(self):
        lines = [
            self.line(0.0, 1.8, "When we got to the station"),
            self.line(2.0, 3.6, "the train had already left."),
            self.line(5.0, 7.0, "That was a long day for everyone."),
            self.line(7.2, 7.8, "Yes."),
            self.line(8.0, 9.5, "Who are you?", speaker="speaker2"),
            self.line(9.6, 11.0, "Nobody you know.", speaker="speaker2"),
        ]
        self.assertEqual(review_checks.combine_suggestions(lines), [[0, 1], [2, 3]])

    def test_limits_keep_lines_apart(self):
        long = [self.line(i * 2.0, i * 2.0 + 1.9, "and then") for i in range(8)]
        groups = review_checks.combine_suggestions(long)
        self.assertTrue(all(long[g[-1]]["end"] - long[g[0]]["start"] <= review_checks.COMBINE_MAX_SECONDS
                            for g in groups))
        gap = [self.line(0, 1, "and then"), self.line(2.5, 3, "we left")]
        self.assertEqual(review_checks.combine_suggestions(gap), [])
        mixed = [self.line(0, 1, "and then"), self.line(1.1, 2, "we left", dub=False)]
        self.assertEqual(review_checks.combine_suggestions(mixed), [])


class DubReviewPageTests(PageTests):
    LINES = [
        {"start": 0.0, "end": 1.5, "speaker": "speaker1", "source": "Quand on est arrivés", "translation": "When we got there"},
        {"start": 1.6, "end": 3.0, "speaker": "speaker1", "source": "le train était parti.", "translation": "the train had left."},
        {"start": 5.0, "end": 7.0, "speaker": "speaker2", "source": "Il fait froid.", "translation": "It is hot."},
        {"start": 7.1, "end": 7.6, "speaker": "speaker2", "source": "Oui.", "translation": "Yes."},
    ]

    def test_detect_then_combine_several_groups(self):
        created = self.session([dict(line) for line in self.LINES], target_language="English")
        detected = self.client.post(f"{API}/combine-suggestions", json={"session": created.id, "context": False}).json()
        groups = detected["groups"]
        self.assertEqual(groups, [[0, 1], [2, 3]])
        self.assertFalse(detected["context"])
        merged = self.client.post(f"{API}/line/merge-groups", json={"session": created.id, "groups": groups}).json()
        lines = merged["session"]["board"]["lines"]
        self.assertEqual([line["translation"] for line in lines], ["When we got there the train had left.", "It is hot. Yes."])
        self.assertIn("Combined 4 lines into 2", merged["status"])
        refused = self.client.post(f"{API}/line/merge-groups", json={"session": created.id, "groups": [[0, 1]]}).json()
        self.assertIn("more than one speaker", refused["error"])

    def test_context_check_joins_lines_whatever_their_speaker_labels(self):
        lines = [dict(line) for line in self.LINES]
        lines[1]["speaker"] = "speaker2"  # the separation split one sentence between two voices
        created = self.session(lines, target_language="English")
        asked = {}

        def decide(questions, context, *, key, **kwargs):
            asked.update(questions)
            return {name: {"type": "noul", "noul": 0.9 if name == "join_0" else 0.2} for name in questions}

        with patch.object(decisions, "key_or_none", return_value="k"), patch.object(decisions, "decide", side_effect=decide):
            detected = self.client.post(f"{API}/combine-suggestions", json={"session": created.id}).json()
        self.assertEqual(set(asked), {"join_0", "join_2"})  # only pairs close in time are asked
        self.assertIn("line 1 (speaker1", asked["join_0"]["instructions"])
        self.assertEqual(detected["groups"], [[0, 1]])
        self.assertTrue(detected["context"])
        refused = self.client.post(f"{API}/line/merge-groups", json={"session": created.id, "groups": [[0, 1]]}).json()
        self.assertIn("more than one speaker", refused["error"])
        merged = self.client.post(f"{API}/line/merge-groups",
                                  json={"session": created.id, "groups": [[0, 1]], "relabel": True}).json()
        first = merged["session"]["board"]["lines"][0]
        self.assertEqual(first["translation"], "When we got there the train had left.")
        self.assertEqual(first["speakerId"], "speaker1")  # heard longest in the group

    def test_context_check_falls_back_to_rules_without_a_key(self):
        created = self.session([dict(line) for line in self.LINES], target_language="English")
        with patch.object(decisions, "key_or_none", return_value=None):
            detected = self.client.post(f"{API}/combine-suggestions", json={"session": created.id}).json()
        self.assertEqual(detected["groups"], [[0, 1], [2, 3]])
        self.assertFalse(detected["context"])
        self.assertIn("No Gemini key", detected["status"])

    def test_translation_check_flags_lines_until_edited(self):
        created = self.session([dict(line) for line in self.LINES], target_language="English")

        def decide(questions, context, *, key, **kwargs):
            return {name: {"type": "noul", "noul": 0.1 if name == "meaning_2" else 0.9} for name in questions}

        with patch.object(decisions, "key_or_none", return_value="k"), patch.object(decisions, "decide", side_effect=decide):
            checked = self.client.post(f"{API}/check-translations", json={"session": created.id}).json()
        self.assertIn("1 to review (line 3)", checked["status"])
        lines = checked["session"]["board"]["lines"]
        self.assertIn("translation", lines[2]["flags"])
        self.assertIn("meaning may differ", lines[2]["check"])
        self.assertEqual(lines[0]["check"], "")
        self.assertIn(2, checked["session"]["readiness"]["issues"])
        line = lines[2]
        saved = self.client.patch(f"{API}/line", json={
            "session": created.id, "key": {"index": 2, "start": line["s"], "speaker": line["speakerId"], "source": line["source"]},
            "start": "0:05.000", "end": "0:07.000", "speaker": line["speakerId"], "original": line["source"],
            "translation": "It is cold.", "dub": True}).json()
        self.assertEqual(saved["session"]["board"]["lines"][2]["check"], "")

    def test_fixing_the_original_clears_a_translation_check(self):
        created = self.session([dict(line) for line in self.LINES], target_language="English")
        decide = lambda questions, context, *, key, **kwargs: {
            name: {"type": "noul", "noul": 0.1 if name == "meaning_2" else 0.9} for name in questions}
        with patch.object(decisions, "key_or_none", return_value="k"), patch.object(decisions, "decide", side_effect=decide):
            line = self.client.post(f"{API}/check-translations", json={"session": created.id}).json()["session"]["board"]["lines"][2]
        saved = self.client.patch(f"{API}/line", json={
            "session": created.id, "key": {"index": 2, "start": line["s"], "speaker": line["speakerId"], "source": line["source"]},
            "start": "0:05.000", "end": "0:07.000", "speaker": line["speakerId"], "original": "Il fait chaud.",
            "translation": line["translation"], "dub": True}).json()
        self.assertEqual(saved["session"]["board"]["lines"][2]["check"], "")

    def test_translation_check_needs_a_key(self):
        created = self.session([dict(line) for line in self.LINES], target_language="English")
        with patch.object(decisions, "key_or_none", return_value=None):
            refused = self.client.post(f"{API}/check-translations", json={"session": created.id}).json()
        self.assertIn("GEMINI_API_KEY", refused["error"])


class MediaReviewTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.image = self.root / "result.png"
        self.image.write_bytes(b"\x89PNG fake")

    def answers(self, questions, blocks, *, key, **kwargs):
        found = {"fit": {"type": "score", "score": 3.0, "legend": {str(i): "" for i in range(5)}}}
        for name, question in questions.items():
            if question["type"] == "choice":
                first = next(iter(question["criteria"]))
                found[name] = {"type": "choice", "choice": "none" if name == "tag_mood" else first,
                               "probabilities": {first: 0.8, "none": 0.9}}
            elif question["type"] == "noul":
                found[name] = {"type": "noul", "noul": 0.9 if name == "tag_animals" else 0.1}
        return found

    def test_one_request_scores_and_tags(self):
        from h3_app import media_review

        with patch.object(decisions, "decide", side_effect=self.answers) as asked:
            payload = media_review.review(self.image, "A close-up of a cat", key="k")
        self.assertEqual(asked.call_count, 1)
        questions, blocks = asked.call_args.args
        self.assertIn("fit", questions)
        self.assertEqual(blocks[1]["type"], "image")
        self.assertEqual((payload["fit"], payload["label"]), (75, "Good fit"))
        self.assertIn("close-up", payload["tags"])
        self.assertIn("animals", payload["tags"])
        self.assertNotIn("text", payload["tags"])
        self.assertNotIn("none", payload["tags"])

    def test_schedule_runs_in_the_background_and_tags_the_library(self):
        from h3_app import media_review

        store = MagicMock()
        with patch.object(decisions, "key_or_none", return_value="k"), \
                patch.object(decisions, "decide", side_effect=self.answers), \
                patch("h3_app.media_catalog.get_media_catalog", return_value=MagicMock(store=store)):
            self.assertTrue(media_review.schedule(self.image, "A cat"))
            for _ in range(100):
                if (media_review.read(self.image) or {}).get("status") == media_review.DONE:
                    break
                time.sleep(0.02)
        shown = media_review.public(self.image)
        self.assertEqual(shown["status"], "done")
        self.assertEqual(shown["fit"], 75)
        store.add_tags.assert_called_once()

    def test_nothing_starts_when_off_or_without_a_key(self):
        from h3_app import media_review

        self.assertFalse(media_review.schedule(self.image, "A cat", enabled=False))
        with patch.object(decisions, "key_or_none", return_value=None):
            self.assertFalse(media_review.schedule(self.image, "A cat"))
        self.assertFalse(media_review.schedule(self.image, "   "))
        self.assertIsNone(media_review.public(self.image))

    def test_a_lost_pending_review_reads_as_failed(self):
        from h3_app import media_review

        media_review._write(self.image, {"status": "pending", "started": time.time() - 3600})
        self.assertEqual(media_review.public(self.image)["status"], "failed")

    def test_library_tags_merge_with_the_viewers_own(self):
        from h3_app.workspace_store import WorkspaceStore

        store = WorkspaceStore(self.root / "store")
        asset_id = store.index_asset(self.image, "Image", {"_media": {}})
        store.annotate(asset_id, tags=["mine"], favorite=True)
        self.assertTrue(store.add_tags(self.image, ["night", "mine"]))
        found = store.search_assets(query="night")
        self.assertEqual(found[0]["tags"], ["mine", "night"])
        self.assertEqual(found[0]["favorite"], 1)
        self.assertFalse(store.add_tags(self.root / "other.png", ["x"]))


if __name__ == "__main__":
    unittest.main()
