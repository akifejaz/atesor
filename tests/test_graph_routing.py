"""Tests for graph routing: init, planner, supervisor, build-fix subgraph."""

import unittest

from src.graph import (
    route_init_to_next,
    route_planner_to_next,
    route_supervisor_to_next,
    route_build_result,
    route_verify_result,
    route_fix_result,
)
from src.state import (
    AgentState,
    BuildStatus,
    ErrorCategory,
    TaskPlan,
    TaskPhase,
    BuildPlan,
    BuildPhase,
    AgentRole,
    create_initial_state,
)


def _base_state(**overrides) -> AgentState:
    s = create_initial_state("https://github.com/a/b.git")
    for k, v in overrides.items():
        setattr(s, k, v)
    return s


class TestRouteInit(unittest.TestCase):
    def test_successful_init_routes_to_planner(self):
        s = _base_state(build_status=BuildStatus.PENDING)
        self.assertEqual(route_init_to_next(s), "planner_node")

    def test_failed_init_routes_to_escalate(self):
        s = _base_state(build_status=BuildStatus.FAILED)
        self.assertEqual(route_init_to_next(s), "escalate_node")


class TestRoutePlanner(unittest.TestCase):
    def test_planner_with_plan_routes_to_scout_aggregator(self):
        tp = TaskPlan(
            phases=[TaskPhase(1, "test", "test", AgentRole.SCOUT, False)]
        )
        s = _base_state(task_plan=tp)
        self.assertEqual(route_planner_to_next(s), "scout_aggregator")

    def test_planner_no_plan_escalates(self):
        s = _base_state(task_plan=None)
        self.assertEqual(route_planner_to_next(s), "escalate_node")

    def test_planner_empty_phases_escalates(self):
        tp = TaskPlan(phases=[])
        s = _base_state(task_plan=tp)
        self.assertEqual(route_planner_to_next(s), "escalate_node")


class TestRouteSupervisor(unittest.TestCase):
    def test_success_routes_to_finish(self):
        s = _base_state(build_status=BuildStatus.SUCCESS)
        self.assertEqual(route_supervisor_to_next(s), "finish_node")

    def test_max_attempts_escalates(self):
        s = _base_state(
            attempt_count=5, max_attempts=5, build_status=BuildStatus.FAILED
        )
        self.assertEqual(route_supervisor_to_next(s), "escalate_node")

    def test_error_loop_escalates(self):
        from src.state import ErrorRecord, ErrorCategory
        err = ErrorRecord(message="fail", category=ErrorCategory.COMPILATION)
        s = _base_state(
            attempt_count=3,
            max_attempts=5,
            error_history=[err, err, err],
            build_status=BuildStatus.FAILED,
        )
        self.assertEqual(route_supervisor_to_next(s), "escalate_node")

    def test_missing_deps_routes_to_scout(self):
        from src.state import ErrorRecord
        err = ErrorRecord(
            message="pkg not found",
            category=ErrorCategory.DEPENDENCY,
        )
        tp = TaskPlan(
            phases=[TaskPhase(1, "t", "t", AgentRole.SCOUT, False)]
        )
        s = _base_state(
            task_plan=tp,
            build_status=BuildStatus.FAILED,
            last_error_category=ErrorCategory.DEPENDENCY,
            error_history=[err],
        )
        self.assertEqual(route_supervisor_to_next(s), "scout_node")

    def test_no_task_plan_routes_to_planner(self):
        s = _base_state(task_plan=None, build_status=BuildStatus.PENDING)
        self.assertEqual(route_supervisor_to_next(s), "planner_node")

    def test_pending_build_routes_to_subgraph(self):
        bp = BuildPlan(
            build_system="make",
            build_system_confidence=0.8,
            phases=[BuildPhase(1, "build", ["make"])],
            total_estimated_duration="1m",
        )
        s = _base_state(
            task_plan=TaskPlan(
                phases=[TaskPhase(1, "t", "t", AgentRole.SCOUT, False)]
            ),
            build_plan=bp,
            build_status=BuildStatus.PENDING,
        )
        self.assertEqual(route_supervisor_to_next(s), "build_fix_subgraph")


class TestBuildFixSubgraphRouting(unittest.TestCase):
    def test_build_success_routes_to_verify(self):
        s = _base_state(build_status=BuildStatus.SUCCESS)
        self.assertEqual(route_build_result(s), "verify_node")

    def test_build_fail_routes_to_fix(self):
        s = _base_state(build_status=BuildStatus.FAILED)
        self.assertEqual(route_build_result(s), "fix_node")

    def test_verify_success_exits_subgraph(self):
        s = _base_state(build_status=BuildStatus.SUCCESS)
        self.assertEqual(route_verify_result(s), "__end__")

    def test_verify_fail_routes_to_fix(self):
        s = _base_state(build_status=BuildStatus.FAILED)
        self.assertEqual(route_verify_result(s), "fix_node")

    def test_fix_success_retries_build(self):
        s = _base_state(build_status=BuildStatus.PENDING)
        self.assertEqual(route_fix_result(s), "build_node")

    def test_fix_still_failed_exits_subgraph(self):
        s = _base_state(build_status=BuildStatus.FAILED)
        self.assertEqual(route_fix_result(s), "__end__")


if __name__ == "__main__":
    unittest.main()
