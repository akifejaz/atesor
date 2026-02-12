import unittest
from datetime import datetime
from src.state import (
    AgentState, 
    BuildStatus, 
    ErrorCategory, 
    create_initial_state, 
    classify_error, 
    create_error_record,
    ErrorRecord
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

if __name__ == "__main__":
    unittest.main()
