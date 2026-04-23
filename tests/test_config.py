"""Tests for src/config.py – environment detection and path management."""

import io
import os
import unittest
from unittest.mock import patch, mock_open, MagicMock

from src import config
from src.config import (
    is_running_in_docker,
    get_workspace_root,
    get_output_dir,
    get_repos_dir,
    get_cache_dir,
    get_logs_dir,
    print_config,
    CONTAINER_NAME,
    IMAGE_NAME,
)


# ---------------------------------------------------------------------------
# is_running_in_docker
# ---------------------------------------------------------------------------
class TestIsRunningInDocker(unittest.TestCase):
    """Test Docker environment detection."""

    @patch("src.config.os.path.exists", return_value=True)
    def test_dockerenv_file_exists(self, mock_exists):
        """Return True when /.dockerenv exists."""
        self.assertTrue(is_running_in_docker())
        mock_exists.assert_any_call("/.dockerenv")

    @patch("src.config.os.path.exists", return_value=False)
    @patch("builtins.open", mock_open(read_data="1:cpu:/docker/abc123\n"))
    def test_cgroup_contains_docker(self, mock_exists):
        """Return True when /proc/1/cgroup mentions 'docker'."""
        self.assertTrue(is_running_in_docker())

    @patch("src.config.os.path.exists", return_value=False)
    @patch("builtins.open", mock_open(read_data="1:cpu:/containerd/abc\n"))
    def test_cgroup_contains_containerd(self, mock_exists):
        """Return True when /proc/1/cgroup mentions 'containerd'."""
        self.assertTrue(is_running_in_docker())

    @patch("src.config.os.path.exists", return_value=False)
    @patch("builtins.open", mock_open(read_data="1:cpu:/kubepods/pod-xyz\n"))
    def test_cgroup_contains_kubepods(self, mock_exists):
        """Return True when /proc/1/cgroup mentions 'kubepods'."""
        self.assertTrue(is_running_in_docker())

    @patch("src.config.os.path.exists", return_value=False)
    @patch("builtins.open", side_effect=FileNotFoundError)
    def test_cgroup_file_not_found(self, mock_file, mock_exists):
        """Gracefully handle missing /proc/1/cgroup (FileNotFoundError)."""
        # Should not raise; falls through to heuristic check (returns False
        # because os.path.exists is always False here).
        self.assertFalse(is_running_in_docker())

    @patch("src.config.os.path.exists", return_value=False)
    @patch("builtins.open", side_effect=PermissionError)
    def test_cgroup_permission_error(self, mock_file, mock_exists):
        """Gracefully handle PermissionError reading cgroup."""
        self.assertFalse(is_running_in_docker())

    @patch("src.config.os.path.exists", return_value=False)
    @patch("builtins.open", mock_open(read_data="1:cpu:/\n"))
    def test_not_in_docker(self, mock_exists):
        """Return False when no Docker indicators are present."""
        self.assertFalse(is_running_in_docker())

    @patch("src.config.os.path.abspath", return_value="/")
    @patch("builtins.open", side_effect=FileNotFoundError)
    def test_heuristic_workspace_no_home(self, mock_file, mock_abspath):
        """Return True via heuristic: /workspace exists, /home does not."""
        def exists_side_effect(path):
            if path == "/.dockerenv":
                return False
            if path == "/workspace":
                return True
            if path == "/home":
                return False
            return False

        with patch("src.config.os.path.exists", side_effect=exists_side_effect):
            self.assertTrue(is_running_in_docker())


# ---------------------------------------------------------------------------
# get_workspace_root
# ---------------------------------------------------------------------------
class TestGetWorkspaceRoot(unittest.TestCase):
    """Test workspace root resolution."""

    @patch("src.config.is_running_in_docker", return_value=True)
    def test_docker_workspace(self, _mock):
        """In Docker the workspace root is /workspace."""
        self.assertEqual(get_workspace_root(), "/workspace")

    @patch("src.config.is_running_in_docker", return_value=False)
    def test_host_workspace(self, _mock):
        """On the host the workspace root is <project>/workspace."""
        result = get_workspace_root()
        self.assertTrue(result.endswith("workspace"))
        self.assertTrue(os.path.isabs(result))


# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------
class TestDirectoryHelpers(unittest.TestCase):
    """Test get_output_dir, get_repos_dir, get_cache_dir, get_logs_dir."""

    @patch("src.config.get_workspace_root", return_value="/fake/workspace")
    @patch("src.config.os.makedirs")
    def test_get_output_dir(self, mock_makedirs, _ws):
        result = get_output_dir()
        self.assertEqual(result, "/fake/workspace/output")
        mock_makedirs.assert_called_once_with("/fake/workspace/output", exist_ok=True)

    @patch("src.config.get_workspace_root", return_value="/fake/workspace")
    @patch("src.config.os.makedirs")
    def test_get_repos_dir(self, mock_makedirs, _ws):
        result = get_repos_dir()
        self.assertEqual(result, "/fake/workspace/repos")
        mock_makedirs.assert_called_once_with("/fake/workspace/repos", exist_ok=True)

    @patch("src.config.get_workspace_root", return_value="/fake/workspace")
    @patch("src.config.os.makedirs")
    def test_get_cache_dir(self, mock_makedirs, _ws):
        result = get_cache_dir()
        self.assertEqual(result, "/fake/workspace/.cache")
        mock_makedirs.assert_called_once_with("/fake/workspace/.cache", exist_ok=True)

    @patch("src.config.get_workspace_root", return_value="/fake/workspace")
    @patch("src.config.os.makedirs")
    def test_get_logs_dir(self, mock_makedirs, _ws):
        result = get_logs_dir()
        self.assertEqual(result, "/fake/workspace/logs")
        mock_makedirs.assert_called_once_with("/fake/workspace/logs", exist_ok=True)


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
class TestModuleConstants(unittest.TestCase):
    """Verify module-level constants."""

    def test_container_name(self):
        self.assertEqual(CONTAINER_NAME, "atesor-ai-sandbox")

    def test_image_name(self):
        self.assertEqual(IMAGE_NAME, "atesor-ai-sandbox:latest")

    def test_workspace_root_is_string(self):
        self.assertIsInstance(config.WORKSPACE_ROOT, str)

    def test_output_dir_is_string(self):
        self.assertIsInstance(config.OUTPUT_DIR, str)

    def test_repos_dir_is_string(self):
        self.assertIsInstance(config.REPOS_DIR, str)

    def test_cache_dir_is_string(self):
        self.assertIsInstance(config.CACHE_DIR, str)

    def test_logs_dir_is_string(self):
        self.assertIsInstance(config.LOGS_DIR, str)

    def test_in_docker_is_bool(self):
        self.assertIsInstance(config._IN_DOCKER, bool)


# ---------------------------------------------------------------------------
# print_config
# ---------------------------------------------------------------------------
class TestPrintConfig(unittest.TestCase):
    """Test print_config output."""

    @patch("sys.stdout", new_callable=io.StringIO)
    def test_print_config_runs(self, mock_stdout):
        """print_config writes expected labels to stdout."""
        print_config()
        output = mock_stdout.getvalue()
        self.assertIn("[Config] Environment:", output)
        self.assertIn("[Config] Workspace:", output)
        self.assertIn("[Config] Output:", output)
        self.assertIn("[Config] Repos:", output)


if __name__ == "__main__":
    unittest.main()
