import unittest
from datetime import datetime
from src.state import (
    AgentState, 
    BuildStatus, 
    ErrorCategory, 
    FailureSeverity,
    create_initial_state, 
    classify_error, 
    create_error_record,
    infer_failure_severity,
    ErrorRecord,
    FixAttempt,
    CommandResult,
    Action,
    AgentRole,
    should_escalate,
    get_next_action_recommendation,
    TaskPlan,
    TaskPhase,
    BuildPlan,
    BuildPhase,
)

class TestState(unittest.TestCase):
    def setUp(self):
        self.repo_url = "https://github.com/example/repo"
        self.state = create_initial_state(self.repo_url)

    def test_initial_state(self):
        self.assertEqual(self.state.repo_url, self.repo_url)
        self.assertEqual(self.state.repo_name, "repo")
        self.assertEqual(self.state.build_status, BuildStatus.PENDING)
        self.assertEqual(self.state.attempt_count, 0)

    def test_add_error(self):
        error = create_error_record("Permission denied", ErrorCategory.PERMISSION)
        self.state.add_error(error)
        
        self.assertEqual(self.state.last_error, "Permission denied")
        self.assertEqual(self.state.last_error_category, ErrorCategory.PERMISSION)
        self.assertEqual(self.state.last_error_severity, FailureSeverity.HIGH)
        self.assertEqual(len(self.state.error_history), 1)
        self.assertEqual(self.state.attempt_count, 1)

    def test_classify_error(self):
        self.assertEqual(classify_error("network timeout"), ErrorCategory.NETWORK)
        self.assertEqual(classify_error("cannot find -lssl"), ErrorCategory.LINKING)
        self.assertEqual(classify_error("undefined reference to 'main'"), ErrorCategory.COMPILATION)
        self.assertEqual(classify_error("cmake: not found"), ErrorCategory.MISSING_TOOLS)
        self.assertEqual(classify_error("unsupported instruction vsetvli"), ErrorCategory.ARCHITECTURE)

    def test_state_serialization(self):
        self.state.log_api_call(cost=0.01)
        state_dict = self.state.to_dict()
        
        self.assertEqual(state_dict['repo_name'], "repo")
        self.assertEqual(state_dict['api_cost_usd'], 0.01)
        self.assertIsInstance(state_dict['created_at'], str)

    def test_error_loop_detection(self):
        for _ in range(3):
            self.state.add_error(create_error_record("Error", ErrorCategory.COMPILATION))
        
        self.assertTrue(self.state.is_in_error_loop())

    def test_infer_failure_severity_levels(self):
        self.assertEqual(
            infer_failure_severity(ErrorCategory.MISSING_TOOLS, command="which ninja"),
            FailureSeverity.LOW,
        )
        self.assertEqual(
            infer_failure_severity(
                ErrorCategory.NETWORK,
                command="git clone --depth 1 https://example.com/repo.git /workspace/repos/repo",
            ),
            FailureSeverity.HIGH,
        )
        self.assertEqual(
            infer_failure_severity(
                ErrorCategory.COMPILATION,
                command="make -j4",
                message="error: undefined reference",
            ),
            FailureSeverity.MEDIUM,
        )

class TestStateExtended(unittest.TestCase):
    """Tests for AgentState methods not covered by TestState."""

    def setUp(self):
        self.state = create_initial_state("https://github.com/example/repo")

    # -- add_fix_attempt --
    def test_add_fix_attempt(self):
        fix = FixAttempt(
            error_category=ErrorCategory.COMPILATION,
            strategy="patch header",
            changes_made=["fixed include"],
            success=True,
        )
        before = self.state.last_updated
        self.state.add_fix_attempt(fix)
        self.assertEqual(len(self.state.fixes_attempted), 1)
        self.assertIs(self.state.fixes_attempted[0], fix)
        self.assertGreaterEqual(self.state.last_updated, before)

    def test_add_multiple_fix_attempts(self):
        for i in range(3):
            self.state.add_fix_attempt(
                FixAttempt(
                    error_category=ErrorCategory.DEPENDENCY,
                    strategy=f"strategy-{i}",
                    changes_made=[],
                    success=i == 2,
                )
            )
        self.assertEqual(len(self.state.fixes_attempted), 3)
        self.assertTrue(self.state.fixes_attempted[2].success)

    # -- log_api_call --
    def test_log_api_call_increments(self):
        self.state.log_api_call(cost=0.05)
        self.state.log_api_call(cost=0.10)
        self.assertEqual(self.state.api_calls_made, 2)
        self.assertAlmostEqual(self.state.api_cost_usd, 0.15)

    def test_log_api_call_default_cost(self):
        self.state.log_api_call()
        self.assertEqual(self.state.api_calls_made, 1)
        self.assertAlmostEqual(self.state.api_cost_usd, 0.0)

    # -- log_scripted_op --
    def test_log_scripted_op(self):
        self.state.log_scripted_op("install_deps")
        self.assertEqual(self.state.scripted_ops_count, 1)
        self.assertEqual(len(self.state.audit_trail), 1)
        self.assertEqual(self.state.audit_trail[0]["event"], "scripted_op")
        self.assertEqual(self.state.audit_trail[0]["data"]["operation"], "install_deps")

    def test_log_scripted_op_default_operation(self):
        self.state.log_scripted_op()
        self.assertEqual(self.state.audit_trail[0]["data"]["operation"], "unknown")

    # -- log_event --
    def test_log_event_structure(self):
        self.state.current_agent = AgentRole.BUILDER
        self.state.log_event("build_start", {"phase": 1})
        entry = self.state.audit_trail[0]
        self.assertEqual(entry["event"], "build_start")
        self.assertEqual(entry["agent"], "builder")
        self.assertEqual(entry["data"], {"phase": 1})
        self.assertIn("timestamp", entry)

    def test_log_event_no_agent(self):
        self.state.current_agent = None
        self.state.log_event("init", {})
        self.assertIsNone(self.state.audit_trail[0]["agent"])

    # -- log_agent_decision --
    def test_log_agent_decision(self):
        self.state.log_agent_decision(AgentRole.SCOUT, "scan_deps", "need dependency info")
        entry = self.state.audit_trail[0]
        self.assertEqual(entry["event"], "decision")
        self.assertEqual(entry["data"]["agent"], "scout")
        self.assertEqual(entry["data"]["action"], "scan_deps")
        self.assertEqual(entry["data"]["reason"], "need dependency info")

    # -- cache_command_result / get_cached_command_result --
    def test_cache_and_retrieve_command_result(self):
        result = CommandResult(
            command="make -j4",
            exit_code=0,
            stdout="OK",
            stderr="",
            duration_seconds=1.5,
        )
        self.state.cache_command_result("make -j4", result)
        cached = self.state.get_cached_command_result("make -j4")
        self.assertIs(cached, result)

    def test_get_cached_command_result_miss(self):
        self.assertIsNone(self.state.get_cached_command_result("nonexistent"))

    def test_cache_command_result_overwrite(self):
        r1 = CommandResult("ls", 0, "a", "", 0.1)
        r2 = CommandResult("ls", 0, "b", "", 0.2)
        self.state.cache_command_result("ls", r1)
        self.state.cache_command_result("ls", r2)
        self.assertIs(self.state.get_cached_command_result("ls"), r2)

    # -- cache_file_content --
    def test_cache_file_content(self):
        before = self.state.last_updated
        self.state.cache_file_content("/src/main.c", "#include <stdio.h>")
        self.assertEqual(self.state.file_content_cache["/src/main.c"], "#include <stdio.h>")
        self.assertGreaterEqual(self.state.last_updated, before)

    # -- get_execution_duration --
    def test_get_execution_duration(self):
        duration = self.state.get_execution_duration()
        self.assertGreaterEqual(duration, 0.0)

    # -- add_build_artifact --
    def test_add_build_artifact(self):
        self.state.add_build_artifact("/out/lib.so", "library", "RISC-V")
        self.assertEqual(len(self.state.build_artifacts), 1)
        art = self.state.build_artifacts[0]
        self.assertEqual(art["filepath"], "/out/lib.so")
        self.assertEqual(art["type"], "library")
        self.assertEqual(art["architecture"], "RISC-V")
        self.assertIn("timestamp", art)

    def test_add_build_artifact_no_arch(self):
        self.state.add_build_artifact("/out/bin", "binary")
        self.assertIsNone(self.state.build_artifacts[0]["architecture"])

    # -- update_timestamp --
    def test_update_timestamp(self):
        old = self.state.last_updated
        self.state.update_timestamp()
        self.assertGreaterEqual(self.state.last_updated, old)


class TestShouldEscalate(unittest.TestCase):
    """Tests for the should_escalate() helper."""

    def setUp(self):
        self.state = create_initial_state("https://github.com/example/repo")

    def test_no_escalation_fresh_state(self):
        esc, reason = should_escalate(self.state)
        self.assertFalse(esc)
        self.assertEqual(reason, "")

    def test_escalate_max_attempts(self):
        self.state.attempt_count = 5
        esc, reason = should_escalate(self.state)
        self.assertTrue(esc)
        self.assertIn("Maximum attempts", reason)

    def test_escalate_error_loop(self):
        for _ in range(3):
            self.state.add_error(create_error_record("same error", ErrorCategory.LINKING))
        esc, reason = should_escalate(self.state)
        self.assertTrue(esc)
        self.assertIn("error loop", reason)

    def test_escalate_license_incompatible(self):
        self.state.last_error_category = ErrorCategory.LICENSE_INCOMPATIBLE
        esc, reason = should_escalate(self.state)
        self.assertTrue(esc)
        self.assertIn("Fundamental blocker", reason)

    def test_escalate_requires_hardware(self):
        self.state.last_error_category = ErrorCategory.REQUIRES_HARDWARE
        esc, reason = should_escalate(self.state)
        self.assertTrue(esc)

    def test_escalate_architecture_impossible(self):
        self.state.last_error_category = ErrorCategory.ARCHITECTURE_IMPOSSIBLE
        esc, reason = should_escalate(self.state)
        self.assertTrue(esc)

    def test_escalate_cost_limit(self):
        self.state.api_cost_usd = 1.50
        esc, reason = should_escalate(self.state)
        self.assertTrue(esc)
        self.assertIn("cost limit", reason)

    def test_no_escalate_below_cost_limit(self):
        self.state.api_cost_usd = 0.50
        esc, _ = should_escalate(self.state)
        self.assertFalse(esc)


class TestGetNextAction(unittest.TestCase):
    """Tests for get_next_action_recommendation()."""

    def setUp(self):
        self.state = create_initial_state("https://github.com/example/repo")

    def test_no_plan_returns_plan(self):
        self.state.task_plan = None
        self.assertEqual(get_next_action_recommendation(self.state), Action.PLAN)

    def test_no_build_plan_returns_scout(self):
        self.state.task_plan = TaskPlan(
            phases=[TaskPhase(id=1, name="build", description="build it",
                              agent=AgentRole.BUILDER, use_scripted_ops=False)],
        )
        self.state.build_plan = None
        self.assertEqual(get_next_action_recommendation(self.state), Action.SCOUT)

    def _set_plans(self):
        """Helper to set both task_plan and build_plan."""
        self.state.task_plan = TaskPlan(
            phases=[TaskPhase(id=1, name="build", description="d",
                              agent=AgentRole.BUILDER, use_scripted_ops=False)],
        )
        self.state.build_plan = BuildPlan(
            build_system="cmake",
            build_system_confidence=0.9,
            phases=[BuildPhase(id=1, name="configure", commands=["cmake ."])],
            total_estimated_duration="5m",
        )

    def test_pending_returns_builder(self):
        self._set_plans()
        self.state.build_status = BuildStatus.PENDING
        self.assertEqual(get_next_action_recommendation(self.state), Action.BUILDER)

    def test_failed_dependency_returns_scout(self):
        self._set_plans()
        self.state.build_status = BuildStatus.FAILED
        self.state.last_error_category = ErrorCategory.DEPENDENCY
        self.assertEqual(get_next_action_recommendation(self.state), Action.SCOUT)

    def test_failed_other_returns_fixer(self):
        self._set_plans()
        self.state.build_status = BuildStatus.FAILED
        self.state.last_error_category = ErrorCategory.COMPILATION
        self.assertEqual(get_next_action_recommendation(self.state), Action.FIXER)

    def test_success_no_tests_returns_builder(self):
        self._set_plans()
        self.state.build_status = BuildStatus.SUCCESS
        self.state.tests_run = False
        self.assertEqual(get_next_action_recommendation(self.state), Action.BUILDER)

    def test_success_with_tests_returns_finish(self):
        self._set_plans()
        self.state.build_status = BuildStatus.SUCCESS
        self.state.tests_run = True
        self.assertEqual(get_next_action_recommendation(self.state), Action.FINISH)

    def test_escalate_overrides(self):
        self._set_plans()
        self.state.attempt_count = 10
        self.assertEqual(get_next_action_recommendation(self.state), Action.ESCALATE)

    def test_default_returns_builder(self):
        self._set_plans()
        self.state.build_status = BuildStatus.BUILDING
        self.assertEqual(get_next_action_recommendation(self.state), Action.BUILDER)


class TestClassifyErrorExtended(unittest.TestCase):
    """Additional classify_error() coverage."""

    def test_rate_limit(self):
        self.assertEqual(classify_error("rate limit exceeded"), ErrorCategory.RATE_LIMIT)
        self.assertEqual(classify_error("HTTP 429 too many requests"), ErrorCategory.RATE_LIMIT)
        self.assertEqual(classify_error("quota exceeded for API"), ErrorCategory.RATE_LIMIT)

    def test_configuration(self):
        self.assertEqual(classify_error("configure error: unsupported host"), ErrorCategory.CONFIGURATION)
        self.assertEqual(classify_error("cmake error at CMakeLists.txt"), ErrorCategory.CONFIGURATION)
        self.assertEqual(classify_error("unsupported option --foo"), ErrorCategory.CONFIGURATION)

    def test_dependency(self):
        self.assertEqual(classify_error("package not found: libfoo"), ErrorCategory.DEPENDENCY)
        self.assertEqual(classify_error("module not found"), ErrorCategory.DEPENDENCY)
        self.assertEqual(classify_error("import error: no module named bar"), ErrorCategory.DEPENDENCY)
        self.assertEqual(classify_error("missing dependency libz"), ErrorCategory.DEPENDENCY)

    def test_permission(self):
        self.assertEqual(classify_error("permission denied /usr/local"), ErrorCategory.PERMISSION)
        self.assertEqual(classify_error("access denied to resource"), ErrorCategory.PERMISSION)

    def test_disk_space(self):
        self.assertEqual(classify_error("no space left on device"), ErrorCategory.DISK_SPACE)
        self.assertEqual(classify_error("disk full"), ErrorCategory.DISK_SPACE)

    def test_python_errors_map_to_configuration(self):
        for err in ["KeyError", "IndexError", "AttributeError", "TypeError", "ValueError"]:
            self.assertEqual(classify_error(err), ErrorCategory.CONFIGURATION, msg=err)

    def test_unknown(self):
        self.assertEqual(classify_error("something completely random xyz"), ErrorCategory.UNKNOWN)


class TestIsInErrorLoopExtended(unittest.TestCase):
    """Extended error-loop detection tests."""

    def setUp(self):
        self.state = create_initial_state("https://github.com/example/repo")

    def test_less_than_three_errors_no_loop(self):
        self.state.add_error(create_error_record("err", ErrorCategory.COMPILATION))
        self.state.add_error(create_error_record("err", ErrorCategory.COMPILATION))
        self.assertFalse(self.state.is_in_error_loop())

    def test_mixed_categories_no_loop(self):
        self.state.add_error(create_error_record("a", ErrorCategory.COMPILATION))
        self.state.add_error(create_error_record("b", ErrorCategory.NETWORK))
        self.state.add_error(create_error_record("c", ErrorCategory.DEPENDENCY))
        self.assertFalse(self.state.is_in_error_loop())

    def test_loop_detected_only_on_last_three(self):
        self.state.add_error(create_error_record("x", ErrorCategory.NETWORK))
        for _ in range(3):
            self.state.add_error(create_error_record("y", ErrorCategory.LINKING))
        self.assertTrue(self.state.is_in_error_loop())


class TestCreateErrorRecordExtended(unittest.TestCase):
    """Extended create_error_record() tests."""

    def test_auto_classify(self):
        rec = create_error_record("network timeout during fetch")
        self.assertEqual(rec.category, ErrorCategory.NETWORK)

    def test_auto_severity(self):
        rec = create_error_record("permission denied", ErrorCategory.PERMISSION)
        self.assertEqual(rec.severity, FailureSeverity.HIGH)

    def test_explicit_severity(self):
        rec = create_error_record("some error", ErrorCategory.UNKNOWN, FailureSeverity.LOW)
        self.assertEqual(rec.severity, FailureSeverity.LOW)

    def test_command_and_attempt(self):
        rec = create_error_record("fail", ErrorCategory.COMPILATION, command="gcc main.c", attempt_number=3)
        self.assertEqual(rec.command, "gcc main.c")
        self.assertEqual(rec.attempt_number, 3)

    def test_auto_classify_and_severity(self):
        rec = create_error_record("disk full, out of space")
        self.assertEqual(rec.category, ErrorCategory.DISK_SPACE)
        self.assertEqual(rec.severity, FailureSeverity.HIGH)


if __name__ == "__main__":
    unittest.main()
