"""The Gemini client's retries of transient failures, without network access."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

from h3_app.llm import gemini


def reply(status):
    return SimpleNamespace(ok=status == 200, status_code=status, json=lambda: {"status": status},
                           text="", headers={})


class GenerateRetryTests(unittest.TestCase):
    def session(self, statuses):
        session = gemini.Session("fixture")
        calls = []

        def post(*args, timeout, **kwargs):
            calls.append(timeout)
            return reply(statuses[len(calls) - 1])

        session.http = SimpleNamespace(post=post)
        return session, calls

    def test_a_transient_failure_is_retried_within_the_budget(self):
        session, calls = self.session([503, 200])
        with patch.object(gemini.time, "sleep"):
            self.assertEqual(session.generate("m", [], generation_config={}, timeout=60), {"status": 200})
        self.assertEqual(len(calls), 2)
        self.assertLessEqual(calls[1], 60)

    def test_no_retry_once_the_budget_is_spent(self):
        session, calls = self.session([503, 200])
        clock = iter([0.0, 55.0, 55.0])
        with patch.object(gemini.time, "monotonic", lambda: next(clock)), patch.object(gemini.time, "sleep"), \
                patch.object(gemini, "error", side_effect=lambda response, action: gemini.H3Error("busy")):
            with self.assertRaises(gemini.H3Error):
                session.generate("m", [], generation_config={}, timeout=60)
        # The one request gets only what is left of the budget, and no retry follows.
        self.assertEqual(calls, [5.0])


if __name__ == "__main__":
    unittest.main()
