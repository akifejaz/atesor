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
    evaluate_state_invariants,
    ErrorRecord,
    CommandResult,
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

    def test_command_cache_invalidation(self):
        result = CommandResult(
            command="make",
            exit_code=0,
            stdout="ok",
            stderr="",
            duration_seconds=0.1,
        )
        self.state.cache_command_result("make", result, cwd="/workspace/repos/repo")
        cached = self.state.get_cached_command_result("make", cwd="/workspace/repos/repo")
        self.assertIsNotNone(cached)

        self.state.invalidate_command_cache("test mutation")
        self.assertIsNone(
            self.state.get_cached_command_result("make", cwd="/workspace/repos/repo")
        )
        self.assertGreater(self.state.command_cache_generation, 0)

    def test_state_invariants(self):
        self.state.current_phase = "builder"
        self.state.build_plan = None
        issues = evaluate_state_invariants(self.state)
        self.assertTrue(any("build plan" in i.lower() for i in issues))

if __name__ == "__main__":
    unittest.main()
