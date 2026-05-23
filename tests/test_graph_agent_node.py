"""Tests for the agent_node decorator: rate-limit retry, generic-error escalation."""

import unittest
from unittest import mock

from src.graph import agent_node
from src.state import AgentRole, BuildStatus, create_initial_state


class TestAgentNodeRateLimitRetry(unittest.TestCase):
    """The decorator retries on rate-limit errors with backoff."""

    def _make_node(self, side_effects):
        calls = {"n": 0}

        @agent_node(AgentRole.SCOUT)
        def node(state):
            calls["n"] += 1
            eff = side_effects[calls["n"] - 1]
            if isinstance(eff, Exception):
                raise eff
            state.current_phase = eff
            return state

        return node, calls

    def test_rate_limit_then_success_returns_normal_state(self):
        node, calls = self._make_node([RuntimeError("Rate limit exceeded"), "scouting"])
        state = create_initial_state("https://x/y.git")
        with mock.patch("time.sleep") as msleep:
            out = node(state)
        self.assertEqual(out.current_phase, "scouting")
        self.assertEqual(calls["n"], 2)
        # First retry should sleep ~30s
        msleep.assert_called_with(30)

    def test_three_rate_limits_then_escalate(self):
        node, calls = self._make_node([
            RuntimeError("429 Too Many Requests"),
            RuntimeError("rate limit"),
            RuntimeError("quota exceeded"),
        ])
        state = create_initial_state("https://x/y.git")
        with mock.patch("time.sleep"):
            out = node(state)
        # After exhausting retries the decorator falls through to the generic
        # error handler on the final attempt → escalate
        self.assertEqual(out.build_status, BuildStatus.ESCALATED)
        self.assertEqual(out.current_phase, "escalate")
        self.assertEqual(calls["n"], 3)

    def test_backoff_doubles_each_retry(self):
        node, _ = self._make_node([
            RuntimeError("rate limit"),
            RuntimeError("rate limit"),
            "scouting",
        ])
        with mock.patch("time.sleep") as msleep:
            node(create_initial_state("https://x/y.git"))
        # 30s, then 60s
        calls = [c.args[0] for c in msleep.call_args_list]
        self.assertEqual(calls[:2], [30, 60])

    def test_all_rate_limit_terms_recognised(self):
        terms = [
            "rate limit",
            "429 too many",
            "Too Many Requests",
            "quota exceeded",
            "RESOURCE_EXHAUSTED",
            "402 payment required",
            "Spend limit exceeded",
            "USD spend limit",
        ]
        for term in terms:
            with self.subTest(term=term):
                node, calls = self._make_node([RuntimeError(term), "scouting"])
                with mock.patch("time.sleep"):
                    node(create_initial_state("https://x/y.git"))
                self.assertEqual(calls["n"], 2, f"Did not retry on: {term}")


class TestAgentNodeGenericError(unittest.TestCase):
    """Generic exceptions escalate; they don't bubble."""

    def test_runtime_error_escalates(self):
        @agent_node(AgentRole.BUILDER)
        def node(state):
            raise ValueError("kaboom — invalid foo")

        state = create_initial_state("https://x/y.git")
        out = node(state)
        self.assertEqual(out.build_status, BuildStatus.ESCALATED)
        self.assertEqual(out.current_phase, "escalate")
        self.assertIn("kaboom", out.last_error)
        # Error should be recorded
        self.assertTrue(len(out.error_history) >= 1)

    def test_exception_does_not_propagate(self):
        @agent_node(AgentRole.FIXER)
        def node(state):
            raise KeyError("missing 'plan'")

        try:
            out = node(create_initial_state("https://x/y.git"))
        except Exception as e:  # pragma: no cover - failure case
            self.fail(f"agent_node should swallow exceptions, got {e!r}")
        self.assertEqual(out.current_phase, "escalate")

    def test_current_agent_set_before_invocation(self):
        captured = {}

        @agent_node(AgentRole.PLANNER)
        def node(state):
            captured["role"] = state.current_agent
            return state

        node(create_initial_state("https://x/y.git"))
        self.assertEqual(captured["role"], AgentRole.PLANNER)


if __name__ == "__main__":
    unittest.main()
