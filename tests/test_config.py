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


if __name__ == "__main__":
    unittest.main()
