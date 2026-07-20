"""Tests for src/config.py — workspace + docker detection."""

import os
import unittest
from unittest import mock

import src.config as config


class TestConfigPaths(unittest.TestCase):
    """Tests for ConfigPaths."""

    def test_workspace_root_exists(self) -> None:
        """Test workspace root exists."""
        self.assertTrue(os.path.isdir(config.WORKSPACE_ROOT))

    def test_subdirs_under_workspace(self) -> None:
        """Test subdirs under workspace."""
        for d in [
            config.OUTPUT_DIR,
            config.REPOS_DIR,
            config.CACHE_DIR,
            config.LOGS_DIR,
        ]:
            with self.subTest(d=d):
                self.assertTrue(
                    d.startswith(config.WORKSPACE_ROOT),
                    f"{d} not under workspace",
                )
                self.assertTrue(os.path.isdir(d))

    def test_get_output_dir_creates_directory(self) -> None:
        """Test get output dir creates directory."""
        out = config.get_output_dir()
        self.assertTrue(os.path.isdir(out))

    def test_docker_detection_false_on_normal_host(self) -> None:
        # Local Linux host should not be misdetected as docker
        # (skip if running in actual container)
        """Test docker detection false on normal host."""
        if os.path.exists("/.dockerenv"):
            self.skipTest("running inside docker")
        self.assertFalse(config.is_running_in_docker())

    def test_dockerenv_file_triggers_detection(self) -> None:
        """Test dockerenv file triggers detection."""
        with mock.patch(
            "os.path.exists", side_effect=lambda p: p == "/.dockerenv"
        ):
            self.assertTrue(config.is_running_in_docker())


class TestAtesorHomeOverride(unittest.TestCase):
    """ATESOR_HOME must win everywhere, including inside a container.

    Regression: get_workspace_root() used to short-circuit on docker
    detection BEFORE consulting ATESOR_HOME, silently sending all
    workspace state to /workspace when the packaged CLI ran inside any
    container (caught by packaging/deb/validate_deb.sh step 6).
    """

    def test_atesor_home_wins_inside_docker(self) -> None:
        """The override beats the in-docker /workspace shortcut."""
        import tempfile

        with tempfile.TemporaryDirectory() as home:
            with mock.patch.dict(os.environ, {"ATESOR_HOME": home}):
                with mock.patch.object(
                    config, "is_running_in_docker", return_value=True
                ):
                    root = config.get_workspace_root()
        self.assertEqual(root, os.path.join(home, "workspace"))

    def test_docker_default_without_override(self) -> None:
        """Without the override the sandbox default stays /workspace."""
        env = {
            k: v for k, v in os.environ.items() if k != "ATESOR_HOME"
        }
        with mock.patch.dict(os.environ, env, clear=True):
            with mock.patch.object(
                config, "is_running_in_docker", return_value=True
            ):
                self.assertEqual(config.get_workspace_root(), "/workspace")


if __name__ == "__main__":
    unittest.main()
