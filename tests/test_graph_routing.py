"""Tests for src/graph.py route_next phase matrix and escalation guard."""

import unittest

from langgraph.graph import END

from src.graph import route_next
from src.state import AgentState, BuildStatus, create_initial_state


def _st(phase: str, status: BuildStatus = BuildStatus.PLANNING) -> AgentState:
    """St."""
    s = create_initial_state("https://x/y.git")
    s.current_phase = phase
    s.build_status = status
    return s


class TestRouteNextPhases(unittest.TestCase):
    """Full phase → node matrix from routing_map."""

    CASES = [
        ("initialization", "planner"),
        ("initialized", "planner"),
        ("planning", "supervisor"),
        ("planned", "supervisor"),
        ("scout", "scout_node"),
        ("scouting", "supervisor"),
        ("builder", "builder_node"),
        ("building", "supervisor"),
        ("fixer", "fixer_node"),
        ("fixing", "supervisor"),
        ("escalate", "escalate_node"),
        ("finish", "finish_node"),
    ]

    def test_all_known_phases_route_correctly(self) -> None:
        """Test all known phases route correctly."""
        for phase, expected in self.CASES:
            with self.subTest(phase=phase):
                self.assertEqual(route_next(_st(phase)), expected)

    def test_terminal_phases_route_to_end(self) -> None:
        """Test terminal phases route to end."""
        self.assertEqual(route_next(_st("escalated")), END)
        self.assertEqual(route_next(_st("finished")), END)

    def test_case_insensitive_phase_matching(self) -> None:
        # current_phase.lower() is taken — accept mixed-case phases
        """Test case insensitive phase matching."""
        self.assertEqual(route_next(_st("Scout")), "scout_node")
        self.assertEqual(route_next(_st("BUILDER")), "builder_node")
        self.assertEqual(route_next(_st("Planning")), "supervisor")

    def test_unknown_phase_falls_back_to_supervisor(self) -> None:
        """Test unknown phase falls back to supervisor."""
        self.assertEqual(route_next(_st("does_not_exist")), "supervisor")
        self.assertEqual(route_next(_st("")), "supervisor")
        self.assertEqual(route_next(_st("foobar")), "supervisor")


class TestEscalationGuard(unittest.TestCase):
    """Failed initialization must escalate, never go to planner."""

    def test_failed_initialization_forces_escalation(self) -> None:
        """Test failed initialization forces escalation."""
        st = _st("initialization", BuildStatus.FAILED)
        self.assertEqual(route_next(st), "escalate_node")

    def test_failed_initialized_forces_escalation(self) -> None:
        """Test failed initialized forces escalation."""
        st = _st("initialized", BuildStatus.FAILED)
        self.assertEqual(route_next(st), "escalate_node")

    def test_in_progress_initialization_routes_normally(self) -> None:
        """Test in progress initialization routes normally."""
        st = _st("initialization", BuildStatus.PLANNING)
        self.assertEqual(route_next(st), "planner")

    def test_failed_status_outside_init_still_uses_routing_map(self) -> None:
        """Test failed status outside init still uses routing map."""
        # FAILED in scouting should still go to supervisor (only init
        # is guarded).
        st = _st("scouting", BuildStatus.FAILED)
        self.assertEqual(route_next(st), "supervisor")


if __name__ == "__main__":
    unittest.main()
