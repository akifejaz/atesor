import unittest

from src.graph import route_next, supervisor_node
from src.state import BuildStatus, create_initial_state


class TestGraphRouting(unittest.TestCase):
    def test_failed_initialization_routes_to_escalation(self):
        state = create_initial_state("https://github.com/example/repo")
        state.current_phase = "initialization"
        state.build_status = BuildStatus.FAILED

        self.assertEqual(route_next(state), "escalate_node")

    def test_supervisor_feedback_routes_to_scout_when_build_plan_missing(self):
        state = create_initial_state("https://github.com/example/repo")
        state.current_phase = "builder"
        state.build_status = BuildStatus.PENDING
        state.task_plan = object()  # simulate existing plan
        state.build_plan = None

        updated = supervisor_node(state)
        self.assertEqual(updated.current_phase, "scout")
        self.assertTrue(updated.feedback_notes)


if __name__ == "__main__":
    unittest.main()
