import unittest
from unittest.mock import patch, MagicMock
import os
from src.scripted_ops import ScriptedOperations
from src.state import CommandResult

class TestScriptedOps(unittest.TestCase):
    def setUp(self):
        # Use a temporary workspace path for testing
        self.workspace = "/tmp/test_workspace"
        self.ops = ScriptedOperations(workspace_root=self.workspace)

    @patch('os.path.exists')
    def test_detect_build_system_cmake(self, mock_exists):
        # Mock file discovery
        def side_effect(path):
            return "CMakeLists.txt" in path
        mock_exists.side_effect = side_effect
        
        info = self.ops.detect_build_system("/workspace/repo")
        self.assertEqual(info.type, "cmake")
        self.assertEqual(info.primary_file, "CMakeLists.txt")
        self.assertGreater(info.confidence, 0.9)

    @patch('os.path.exists')
    def test_detect_build_system_cargo(self, mock_exists):
        def side_effect(path):
            return "Cargo.toml" in path
        mock_exists.side_effect = side_effect
        
        info = self.ops.detect_build_system("/workspace/repo")
        self.assertEqual(info.type, "cargo")
        self.assertEqual(info.primary_file, "Cargo.toml")

    @patch('src.scripted_ops.execute_command')
    def test_get_repository_info(self, mock_exec):
        # Mock git commands
        def exec_side_effect(cmd, **kwargs):
            if "rev-parse" in cmd:
                return CommandResult(cmd, 0, "abcdef123", "", 0.1)
            if "branch" in cmd:
                return CommandResult(cmd, 0, "main", "", 0.1)
            if "wc -l" in cmd:
                return CommandResult(cmd, 0, "150", "", 0.1)
            return CommandResult(cmd, 1, "", "Error", 0.1)
            
        mock_exec.side_effect = exec_side_effect
        
        info = self.ops.get_repository_info("/workspace/repo")
        self.assertEqual(info['commit'], "abcdef123")
        self.assertEqual(info['branch'], "main")
        self.assertEqual(info['file_count'], "150")

if __name__ == "__main__":
    unittest.main()
