"""
Build system pattern analyzer for RISC-V porting.
Detects architecture-specific build patterns and generates RISC-V equivalents.
"""

import os
import re
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class BuildPattern:
    pattern_type: str
    arch_files: Dict[str, str]
    riscv_needs_creation: bool
    template_file: Optional[str] = None
    suggested_files: List[Dict[str, str]] = field(default_factory=list)


class BuildSystemAnalyzer:
    """
    Analyzes build system patterns to detect architecture-specific configurations.
    Generates RISC-V build files based on existing patterns.
    """

    ARCH_PATTERNS = {
        "x86_64": ["x64", "x86_64", "amd64", "x86", "i386", "i686"],
        "arm64": ["arm64", "aarch64", "arm", "armv7", "armv8"],
        "riscv": ["riscv", "riscv64", "rv64"],
    }

    BUILD_FILE_PATTERNS = [
        (r"cmpl_gcc_(\w+)\.mak", "gcc_makefile"),
        (r"var_gcc_(\w+)\.mak", "gcc_vars"),
        (r"makefile\.(\w+)", "arch_makefile"),
        (r"(\w+)\.mk", "arch_mk"),
        (r"config\.(\w+)", "config_arch"),
        (r"CMakeLists\.txt", "cmake"),
        (r"configure\.(\w+)", "configure_arch"),
        (r"build_(\w+)\.sh", "build_script"),
    ]

    def __init__(self, repo_path: str):
        self.repo_path = repo_path
        self.detected_patterns: List[BuildPattern] = []

    def analyze_build_patterns(self) -> List[BuildPattern]:
        """
        Analyze the repository for architecture-specific build patterns.
        Returns list of patterns found and whether RISC-V support exists.
        """
        patterns = []

        for root, dirs, files in os.walk(self.repo_path):
            if ".git" in root:
                continue

            for pattern_regex, pattern_type in self.BUILD_FILE_PATTERNS:
                for filename in files:
                    match = re.match(pattern_regex, filename, re.IGNORECASE)
                    if match:
                        arch = match.group(1).lower() if match.groups() else "generic"
                        full_path = os.path.join(root, filename)
                        rel_path = os.path.relpath(full_path, self.repo_path)

                        pattern = self._analyze_pattern(
                            pattern_type, arch, full_path, rel_path
                        )
                        if pattern:
                            patterns.append(pattern)

        self.detected_patterns = self._deduplicate_patterns(patterns)
        return self.detected_patterns

    def _analyze_pattern(
        self, pattern_type: str, arch: str, full_path: str, rel_path: str
    ) -> Optional[BuildPattern]:
        """Analyze a single build file pattern."""

        detected_arch = None
        for arch_name, arch_patterns in self.ARCH_PATTERNS.items():
            if any(p in arch for p in arch_patterns):
                detected_arch = arch_name
                break

        if detected_arch is None:
            return None

        pattern = BuildPattern(
            pattern_type=pattern_type,
            arch_files={detected_arch: rel_path},
            riscv_needs_creation=True,
            template_file=full_path,
        )

        riscv_file = self._generate_riscv_filename(rel_path, detected_arch)
        riscv_full_path = os.path.join(self.repo_path, riscv_file)

        if os.path.exists(riscv_full_path):
            pattern.riscv_needs_creation = False
            pattern.arch_files["riscv"] = riscv_file
        else:
            pattern.suggested_files.append(
                {
                    "source": rel_path,
                    "target": riscv_file,
                    "action": "create_from_template",
                }
            )

        return pattern

    def _generate_riscv_filename(self, original_path: str, source_arch: str) -> str:
        """Generate RISC-V equivalent filename from an architecture-specific file."""

        arch_replacements = {
            "x86_64": ["x64", "x86_64", "amd64"],
            "arm64": ["arm64", "aarch64"],
            "x86": ["x86", "i386", "i686"],
        }

        result = original_path
        for arch_name, patterns in arch_replacements.items():
            if arch_name == source_arch:
                for pattern in patterns:
                    result = re.sub(pattern, "riscv64", result, flags=re.IGNORECASE)
                    result = re.sub(pattern.upper(), "RISCV64", result)

        return result

    def _deduplicate_patterns(self, patterns: List[BuildPattern]) -> List[BuildPattern]:
        """Remove duplicate patterns and merge similar ones."""
        seen = {}
        for pattern in patterns:
            key = pattern.pattern_type
            if key not in seen:
                seen[key] = pattern
            else:
                seen[key].arch_files.update(pattern.arch_files)
                seen[key].suggested_files.extend(pattern.suggested_files)

        return list(seen.values())

    def get_riscv_build_instructions(self) -> Dict[str, Any]:
        """
        Generate instructions for RISC-V build based on detected patterns.
        """
        result = {
            "has_riscv_support": False,
            "needs_file_creation": False,
            "files_to_create": [],
            "build_commands": [],
            "analysis_summary": "",
        }

        riscv_patterns = [
            p for p in self.detected_patterns if not p.riscv_needs_creation
        ]
        creation_patterns = [
            p for p in self.detected_patterns if p.riscv_needs_creation
        ]

        if riscv_patterns:
            result["has_riscv_support"] = True
            for pattern in riscv_patterns:
                if "riscv" in pattern.arch_files:
                    riscv_file = pattern.arch_files["riscv"]
                    if pattern.pattern_type == "gcc_makefile":
                        result["build_commands"].append(f"make -f {riscv_file}")

        if creation_patterns:
            result["needs_file_creation"] = True
            for pattern in creation_patterns:
                for suggestion in pattern.suggested_files:
                    result["files_to_create"].append(suggestion)

        result["analysis_summary"] = self._generate_summary(result)

        return result

    def _generate_summary(self, result: Dict[str, Any]) -> str:
        """Generate human-readable analysis summary."""
        lines = []

        if result["has_riscv_support"]:
            lines.append("RISC-V build support already exists.")
        else:
            lines.append("RISC-V build support NOT found.")

        if result["needs_file_creation"]:
            lines.append(
                f"\nDetected architecture-specific build patterns. "
                f"Need to create {len(result['files_to_create'])} file(s) for RISC-V:"
            )
            for f in result["files_to_create"]:
                lines.append(f"  - Create {f['target']} based on {f['source']}")

        return "\n".join(lines)

    def generate_riscv_build_file(
        self, template_path: str, target_path: str
    ) -> Tuple[bool, str]:
        """
        Generate a RISC-V build file from a template.
        Returns (success, content).
        """
        try:
            with open(template_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            arch_replacements = {
                "x64": "riscv64",
                "x86_64": "riscv64",
                "amd64": "riscv64",
                "X64": "RISCV64",
                "AMD64": "RISCV64",
                "arm64": "riscv64",
                "aarch64": "riscv64",
                "ARM64": "RISCV64",
                "AARCH64": "RISCV64",
                "arm": "riscv64",
                "ARM": "RISCV64",
                "x86": "riscv64",
                "i386": "riscv64",
                "i686": "riscv64",
            }

            result = content
            for old, new in arch_replacements.items():
                result = result.replace(old, new)

            asm_patterns = [
                (r"USE_ASM\s*=\s*1", "USE_ASM ="),
                (r"HAVE_ASM\s*=\s*yes", "HAVE_ASM = no"),
            ]

            for pattern, replacement in asm_patterns:
                result = re.sub(pattern, replacement, result)

            return True, result

        except Exception as e:
            logger.error(f"Failed to generate RISC-V build file: {e}")
            return False, f"Error: {e}"


def analyze_build_system_for_riscv(repo_path: str) -> Dict[str, Any]:
    """
    Main entry point for build system analysis.
    Returns comprehensive analysis for RISC-V porting.
    """
    analyzer = BuildSystemAnalyzer(repo_path)
    patterns = analyzer.analyze_build_patterns()
    instructions = analyzer.get_riscv_build_instructions()

    result = {
        "patterns_found": [
            {
                "type": p.pattern_type,
                "archs": list(p.arch_files.keys()),
                "needs_riscv_creation": p.riscv_needs_creation,
            }
            for p in patterns
        ],
        **instructions,
    }

    if instructions["needs_file_creation"]:
        result["template_contents"] = {}
        for pattern in patterns:
            if pattern.template_file and pattern.riscv_needs_creation:
                success, content = analyzer.generate_riscv_build_file(
                    pattern.template_file, ""
                )
                if success:
                    rel_path = os.path.relpath(pattern.template_file, repo_path)
                    result["template_contents"][rel_path] = content

    return result
