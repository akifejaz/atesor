#############################################################################
# Copyright (c) 2026 10xEngineers
#
# Author: Akif Ejaz <akif.ejaz@10xengineers.ai>
# This program and the accompanying materials are made available under the
# terms of the MIT License which is available at
# https://opensource.org/licenses/MIT.
#
# SPDX-License-Identifier: MIT
#############################################################################

"""Build artifact detection and verification.

Scans build directories for artifacts and verifies that they target the
RISC-V architecture.
"""

import logging
import shlex
import tempfile
from typing import Any, Dict, List, Optional, Tuple

from .tools import execute_command

logger = logging.getLogger(__name__)


class ArtifactScanner:
    """Scan for and verify build artifacts.

    Attributes:
        build_dir: Directory scanned for build artifacts.
        cwd: Working directory used for command execution.
        artifacts: Discovered artifacts with their metadata.
    """

    def __init__(self, build_dir: str, cwd: Optional[str] = None) -> None:
        self.build_dir = build_dir
        self.cwd = cwd or build_dir
        self.artifacts: List[Dict[str, Any]] = []

    def scan(self) -> List[Dict[str, Any]]:
        """Scan the build directory for artifacts.

        Returns:
            A list of discovered artifacts with their metadata.
        """
        self.artifacts = []

        # Find executable binaries.
        find_cmd = (
            f"find {shlex.quote(self.build_dir)} -path '*/.git' -prune"
            " -o -type f -executable -print 2>/dev/null"
            " | grep -v '.so' | head -20"
        )
        binaries_result = execute_command(find_cmd, cwd=self.cwd)
        if binaries_result.success and binaries_result.stdout.strip():
            for binary_path in binaries_result.stdout.strip().split("\n"):
                self._check_artifact(binary_path, "binary")

        # Find static libraries.
        libs_cmd = (
            f"find {shlex.quote(self.build_dir)} -path '*/.git' -prune"
            " -o -name '*.a' -print 2>/dev/null | head -20"
        )
        libs_result = execute_command(libs_cmd, cwd=self.cwd)
        if libs_result.success and libs_result.stdout.strip():
            for lib_path in libs_result.stdout.strip().split("\n"):
                self._check_artifact(lib_path, "library_static")

        # Find shared libraries.
        shared_cmd = (
            f"find {shlex.quote(self.build_dir)} -path '*/.git' -prune"
            " -o -name '*.so*' -print 2>/dev/null | head -20"
        )
        shared_result = execute_command(shared_cmd, cwd=self.cwd)
        if shared_result.success and shared_result.stdout.strip():
            for lib_path in shared_result.stdout.strip().split("\n"):
                self._check_artifact(lib_path, "library_shared")

        logger.info(
            f"Artifact scan complete: {len(self.artifacts)} artifacts found"
        )
        return self.artifacts

    def _check_artifact(self, filepath: str, artifact_type: str) -> None:
        """Check a file and verify its architecture.

        Args:
            filepath: Path to the artifact.
            artifact_type: Type of artifact (binary, library_static, ...).
        """
        file_result = execute_command(
            f"file {shlex.quote(filepath)}", cwd=self.cwd
        )
        if not file_result.success:
            logger.warning(f"Could not get file info for {filepath}")
            return

        file_info = file_result.stdout.strip()
        architecture = self._detect_architecture(file_info)

        # For static libraries, try extracting object files to check arch
        if artifact_type == "library_static" and "ar archive" in file_info:
            arch_from_objects = self._get_archive_architecture(filepath)
            if arch_from_objects:
                architecture = arch_from_objects

        if not architecture:
            logger.warning(
                f"Could not detect architecture for {filepath}: {file_info}"
            )

        artifact = {
            "filepath": filepath,
            "type": artifact_type,
            "architecture": architecture,
            "file_info": file_info,
            "size_bytes": self._get_file_size(filepath),
        }
        self.artifacts.append(artifact)
        logger.info(f"Found {artifact_type} {filepath}")

    def _detect_architecture(self, file_info: str) -> Optional[str]:
        """Detect the architecture from ``file`` command output.

        Args:
            file_info: Output from the ``file`` command.

        Returns:
            An architecture string, or None if undetected.
        """
        file_info_lower = file_info.lower()

        if "risc-v" in file_info_lower or "riscv64" in file_info_lower:
            return "RISC-V"
        if "x86-64" in file_info_lower or "x86_64" in file_info_lower:
            return "x64"
        if "aarch64" in file_info_lower or "arm64" in file_info_lower:
            return "ARM64"
        if "arm" in file_info_lower:
            return "ARM32"
        if "80386" in file_info_lower or "intel 80386" in file_info_lower:
            return "x86"

        return None

    def _get_archive_architecture(self, archive_path: str) -> Optional[str]:
        """Detect architecture from object files inside an archive.

        Args:
            archive_path: Path to the ``.a`` archive.

        Returns:
            An architecture string, or None if undetected.
        """
        tmp_dir = None
        try:
            tmp_dir = tempfile.mkdtemp(prefix="atesor_ar_")
            extract_cmd = (
                f"cd {shlex.quote(tmp_dir)} && ar x"
                f" {shlex.quote(archive_path)} 2>/dev/null"
                " && file *.o | head -1"
            )
            result = execute_command(extract_cmd, cwd=tmp_dir)
            if result.success and result.stdout:
                return self._detect_architecture(result.stdout)
        except Exception as e:
            logger.debug(f"Failed to check archive architecture: {e}")
        finally:
            if tmp_dir:
                import shutil

                shutil.rmtree(tmp_dir, ignore_errors=True)

        return None

    def _get_file_size(self, filepath: str) -> int:
        """Get file size in bytes."""
        try:
            size_result = execute_command(
                f"stat -c %s {shlex.quote(filepath)} 2>/dev/null"
                f" || stat -f %z {shlex.quote(filepath)}",
                cwd=self.cwd,
            )
            if size_result.success:
                return int(size_result.stdout.strip())
        except Exception as e:
            logger.debug(f"Could not get file size for {filepath}: {e}")

        return 0

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of found artifacts."""
        by_type: Dict[str, int] = {}
        by_arch: Dict[str, int] = {}

        for artifact in self.artifacts:
            artifact_type = artifact.get("type", "")
            by_type[artifact_type] = by_type.get(artifact_type, 0) + 1

            architecture = artifact.get("architecture", "")
            if architecture:
                by_arch[architecture] = by_arch.get(architecture, 0) + 1

        return {
            "total_artifacts": len(self.artifacts),
            "by_type": by_type,
            "by_architecture": by_arch,
            "has_riscv": any(
                a.get("architecture") == "RISC-V" for a in self.artifacts
            ),
            "artifacts": self.artifacts,
        }

    def verify_build_success(self) -> Tuple[bool, str]:
        """Verify whether the build produced RISC-V artifacts.

        Returns:
            A ``(is_successful, message)`` tuple.
        """
        summary = self.get_summary()

        if not summary["total_artifacts"]:
            return False, "No build artifacts found in build directory"

        if summary["has_riscv"]:
            type_str = ", ".join(
                f"{k}: {v}" for k, v in summary["by_type"].items()
            )
            total = summary["total_artifacts"]
            message = (
                f"Build successful: {total} RISC-V artifacts found"
                f" ({type_str})"
            )
            return True, message

        if summary["by_architecture"]:
            arches = ", ".join(summary["by_architecture"].keys())
            return (
                False,
                "Build produced artifacts but not for RISC-V"
                f" (found: {arches})",
            )

        return (
            False,
            "Build produced artifacts but could not detect architecture",
        )
