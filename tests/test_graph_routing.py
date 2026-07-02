"""Tests for graph routing: init, planner, supervisor, build-fix subgraph."""

import unittest

from src.graph import (
    route_build_result,
    route_fix_result,
    route_init_to_next,
    route_planner_to_next,
    route_supervisor_to_next,
    route_verify_result,
)
from src.state import (
    AgentRole,
    AgentState,
    BuildPhase,
    BuildPlan,
    BuildStatus,
    ErrorCategory,
    TaskPhase,
    TaskPlan,
    create_initial_state,
)


def _base_state(**overrides) -> AgentState:
    """Build a base AgentState with field overrides."""
    s = create_initial_state("https://github.com/a/b.git")
    for k, v in overrides.items():
        setattr(s, k, v)
    return s


class TestRouteInit(unittest.TestCase):
    """Tests for route_init_to_next."""

    def test_successful_init_routes_to_planner(self):
        """Successful init routes to planner."""
        s = _base_state(build_status=BuildStatus.PENDING)
        self.assertEqual(route_init_to_next(s), "planner_node")

    def test_failed_init_routes_to_escalate(self):
        """Failed init routes to escalate."""
        s = _base_state(build_status=BuildStatus.FAILED)
        self.assertEqual(route_init_to_next(s), "escalate_node")


class TestRoutePlanner(unittest.TestCase):
    """Tests for route_planner_to_next."""

    def test_planner_with_plan_routes_to_scout_chain(self):
        """A valid plan enters the scout chain at the FIRST branch node.

        The aggregator must run AFTER the branches so the fan-in
        actually sees their results (regression: aggregator-first
        wiring meant the default plan was always built from empty
        results and every repo got the 'unknown' discovery plan).
        """
        tp = TaskPlan(
            phases=[TaskPhase(1, "test", "test", AgentRole.SCOUT, False)]
        )
        s = _base_state(task_plan=tp)
        self.assertEqual(route_planner_to_next(s), "scout_build_system")

    def test_planner_no_plan_escalates(self):
        """Planner no plan escalates."""
        s = _base_state(task_plan=None)
        self.assertEqual(route_planner_to_next(s), "escalate_node")

    def test_planner_empty_phases_escalates(self):
        """Planner empty phases escalates."""
        tp = TaskPlan(phases=[])
        s = _base_state(task_plan=tp)
        self.assertEqual(route_planner_to_next(s), "escalate_node")


class TestRouteScoutAggregator(unittest.TestCase):
    """Tests for route_scout_aggregator_to_next."""

    def test_aggregated_plan_routes_to_supervisor(self):
        """Aggregated plan routes to supervisor."""
        from src.graph import route_scout_aggregator_to_next

        bp = BuildPlan(
            build_system="cmake",
            build_system_confidence=0.9,
            phases=[BuildPhase(1, "build", ["make"])],
            total_estimated_duration="1m",
        )
        s = _base_state(build_plan=bp)
        self.assertEqual(route_scout_aggregator_to_next(s), "supervisor_node")

    def test_no_plan_defers_to_llm_scout(self):
        """No plan defers to llm scout."""
        s = _base_state(build_plan=None)
        from src.graph import route_scout_aggregator_to_next

        self.assertEqual(route_scout_aggregator_to_next(s), "scout_node")


class TestRouteSupervisor(unittest.TestCase):
    """Tests for route_supervisor_to_next."""

    def test_success_routes_to_finish(self):
        """Success routes to finish."""
        s = _base_state(build_status=BuildStatus.SUCCESS)
        self.assertEqual(route_supervisor_to_next(s), "finish_node")

    def test_success_at_max_attempts_still_finishes(self):
        """A build that succeeded on its final attempt must FINISH.

        Regression: should_escalate() was consulted before the SUCCESS
        check, so a package whose last fix landed on attempt
        max_attempts was escalated and its recipe thrown away.
        """
        s = _base_state(
            build_status=BuildStatus.SUCCESS,
            attempt_count=5,
            max_attempts=5,
        )
        self.assertEqual(route_supervisor_to_next(s), "finish_node")

    def test_success_over_cost_cap_still_finishes(self):
        """Success over cost cap still finishes."""
        s = _base_state(build_status=BuildStatus.SUCCESS, api_cost_usd=1.5)
        self.assertEqual(route_supervisor_to_next(s), "finish_node")

    def test_max_attempts_escalates(self):
        """Max attempts escalates."""
        s = _base_state(
            attempt_count=5, max_attempts=5, build_status=BuildStatus.FAILED
        )
        self.assertEqual(route_supervisor_to_next(s), "escalate_node")

    def test_error_loop_escalates(self):
        """Error loop escalates."""
        from src.state import ErrorCategory, ErrorRecord

        err = ErrorRecord(message="fail", category=ErrorCategory.COMPILATION)
        s = _base_state(
            attempt_count=3,
            max_attempts=5,
            error_history=[err, err, err],
            build_status=BuildStatus.FAILED,
        )
        self.assertEqual(route_supervisor_to_next(s), "escalate_node")

    def test_missing_deps_routes_to_scout(self):
        """Missing deps routes to scout."""
        from src.state import ErrorRecord

        err = ErrorRecord(
            message="pkg not found",
            category=ErrorCategory.DEPENDENCY,
        )
        tp = TaskPlan(phases=[TaskPhase(1, "t", "t", AgentRole.SCOUT, False)])
        s = _base_state(
            task_plan=tp,
            build_status=BuildStatus.FAILED,
            last_error_category=ErrorCategory.DEPENDENCY,
            error_history=[err],
        )
        self.assertEqual(route_supervisor_to_next(s), "scout_node")

    def test_no_task_plan_routes_to_planner(self):
        """No task plan routes to planner."""
        s = _base_state(task_plan=None, build_status=BuildStatus.PENDING)
        self.assertEqual(route_supervisor_to_next(s), "planner_node")

    def test_pending_build_routes_to_subgraph(self):
        """Pending build routes to subgraph."""
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
    """Tests for the build-fix subgraph routing."""

    def test_build_success_routes_to_verify(self):
        """Build success routes to verify."""
        s = _base_state(build_status=BuildStatus.SUCCESS)
        self.assertEqual(route_build_result(s), "verify_node")

    def test_build_fail_routes_to_fix(self):
        """Build fail routes to fix."""
        s = _base_state(build_status=BuildStatus.FAILED)
        self.assertEqual(route_build_result(s), "fix_node")

    def test_verify_success_exits_subgraph(self):
        """Verify success exits subgraph."""
        s = _base_state(build_status=BuildStatus.SUCCESS)
        self.assertEqual(route_verify_result(s), "__end__")

    def test_verify_fail_routes_to_fix(self):
        """Verify fail routes to fix."""
        s = _base_state(build_status=BuildStatus.FAILED)
        self.assertEqual(route_verify_result(s), "fix_node")

    def test_fix_success_retries_build(self):
        """Fix success retries build."""
        s = _base_state(build_status=BuildStatus.PENDING)
        self.assertEqual(route_fix_result(s), "build_node")

    def test_fix_still_failed_exits_subgraph(self):
        """Fix still failed exits subgraph."""
        s = _base_state(build_status=BuildStatus.FAILED)
        self.assertEqual(route_fix_result(s), "__end__")


if __name__ == "__main__":
    unittest.main()
