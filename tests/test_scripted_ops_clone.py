import unittest
from unittest.mock import patch

from src.scripted_ops import ScriptedOperations
from src.state import CommandResult


class TestCloneUpdateBehavior(unittest.TestCase):
    @patch("src.scripted_ops.execute_command")
    @patch("os.path.exists")
    def test_existing_repo_uses_clean_pull(self, mock_exists, mock_exec):
        mock_exists.return_value = True
        mock_exec.return_value = CommandResult(
            command="git pull",
            exit_code=0,
            stdout="ok",
            stderr="",
            duration_seconds=0.1,
        )

        ops = ScriptedOperations(workspace_root="/tmp/test_workspace")
        ops.clone_or_update_repository("https://github.com/example/repo.git", "repo")

        called = "\n".join(call.args[0] for call in mock_exec.call_args_list)
        self.assertIn("git reset --hard HEAD", called)
        self.assertIn("git clean -fd", called)
        self.assertIn("git pull --ff-only", called)


if __name__ == "__main__":
    unittest.main()
