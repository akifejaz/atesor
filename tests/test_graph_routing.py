import unittest

from src.graph import route_next
from src.state import BuildStatus, create_initial_state


class TestGraphRouting(unittest.TestCase):
    def test_failed_initialization_routes_to_escalation(self):
        state = create_initial_state("https://github.com/example/repo")
        state.current_phase = "initialization"
        state.build_status = BuildStatus.FAILED

        self.assertEqual(route_next(state), "escalate_node")


if __name__ == "__main__":
    unittest.main()
