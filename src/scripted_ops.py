"""
Deterministic, non-LLM operations for repository analysis and file management.
Handles repository cloning, build system detection, and dependency extraction.
"""

import os
import re
import json
import subprocess
from typing import List, Dict, Optional, Any
import logging
from .config import WORKSPACE_ROOT, REPOS_DIR, CACHE_DIR

from src.state import (
    BuildSystemInfo,
    DependencyInfo,
    ArchSpecificCode,
    CommunityPortStatus,
    CommandResult,
)
from src.tools import execute_command

logger = logging.getLogger(__name__)


# ============================================================================
# SCRIPTED OPERATIONS CLASS
# ============================================================================


class ScriptedOperations:
    """
    Handles all deterministic operations that don't require LLM intelligence.
    """

    def __init__(self, workspace_root: str = None):
        """Initialize with environment-aware workspace."""
        if workspace_root is None:
            self.workspace_root = WORKSPACE_ROOT
            self.repos_dir = REPOS_DIR
            self.cache_dir = CACHE_DIR
        else:
            self.workspace_root = workspace_root
            self.repos_dir = os.path.join(workspace_root, "repos")
            self.cache_dir = os.path.join(workspace_root, ".cache")
            try:
                os.makedirs(self.repos_dir, exist_ok=True)
                os.makedirs(self.cache_dir, exist_ok=True)
            except PermissionError as e:
                logger.error(f"Permission denied: {e}")
                raise

        logger.info(f"ScriptedOperations initialized: {self.workspace_root}")

    def _to_host_path(self, path: str) -> str:
        """Translate container path to host path if necessary."""
        if path.startswith("/workspace") and not os.path.exists("/workspace"):
            return path.replace("/workspace", self.workspace_root)
        return path

    def _to_container_path(self, path: str) -> str:
        """Translate host path to container path if necessary."""
        if path.startswith(self.workspace_root):
            return path.replace(self.workspace_root, "/workspace")
        return path

    def clone_or_update_repository(self, url: str, name: str) -> CommandResult:
        """
        Clone a repository or update if it already exists.
        This is a zero-cost operation.

        IMPORTANT: Clones inside Docker container to avoid ownership issues.
        """
        # Use container path
        container_repo_path = f"/workspace/repos/{name}"
        host_repo_path = os.path.join(self.repos_dir, name)

        # Check if already exists (check on host for speed)
        if os.path.exists(os.path.join(host_repo_path, ".git")):
            logger.info(f"Repository {name} already exists, pulling latest...")
            # Pull inside container to maintain ownership
            cmd = f"cd {container_repo_path} && git pull"
            result = execute_command(cmd, use_docker=True)
        else:
            logger.info(f"Cloning repository {url}...")
            # Clone inside container
            cmd = f"git clone --depth 1 {url} {container_repo_path}"
            result = execute_command(cmd, use_docker=True)

        # Configure git safe.directory to prevent "dubious ownership" errors
        # This is needed because the workspace is mounted from host
        if result.success or os.path.exists(host_repo_path):
            safe_dir_cmd = (
                f"git config --global --add safe.directory {container_repo_path}"
            )
            safe_result = execute_command(safe_dir_cmd, use_docker=True)
            if not safe_result.success:
                logger.warning(f"Failed to add safe.directory: {safe_result.stderr}")

        return result

    def get_repository_info(self, repo_path: str) -> Dict[str, str]:
        """Get basic repository information."""
        # Use container path for git operations
        container_path = self._to_container_path(repo_path)
        info = {}

        # Get current commit
        result = execute_command(
            f"cd {container_path} && git rev-parse HEAD", use_docker=True
        )
        if result.success:
            info["commit"] = result.stdout.strip()

        # Get branch
        result = execute_command(
            f"cd {container_path} && git branch --show-current", use_docker=True
        )
        if result.success:
            info["branch"] = result.stdout.strip()

        # Count files
        result = execute_command(
            f"find {container_path} -type f | wc -l", use_docker=True
        )
        if result.success:
            info["file_count"] = result.stdout.strip()

        return info

    # ========== Build System Detection ==========

    def detect_build_system(self, repo_path: str) -> BuildSystemInfo:
        """
        Detect build system using file-based heuristics.
        This is a zero-cost operation.
        """
        repo_path = self._to_host_path(repo_path)
        build_systems = {
            "cmake": ["CMakeLists.txt"],
            "autotools": ["configure.ac", "configure.in"],
            "make": ["Makefile", "GNUmakefile"],
            "meson": ["meson.build"],
            "cargo": ["Cargo.toml"],
            "npm": ["package.json"],
            "pip": ["setup.py", "pyproject.toml"],
            "go": ["go.mod"],
            "gradle": ["build.gradle", "build.gradle.kts"],
            "maven": ["pom.xml"],
            "bazel": ["BUILD", "BUILD.bazel", "WORKSPACE"],
        }

        detected = []
        confidence_scores = {}

        for build_sys, files in build_systems.items():
            for file in files:
                filepath = os.path.join(repo_path, file)
                if os.path.exists(filepath):
                    detected.append((build_sys, file))
                    if file in [
                        "CMakeLists.txt",
                        "Cargo.toml",
                        "configure.ac",
                        "go.mod",
                    ]:
                        confidence_scores[build_sys] = 0.95
                    else:
                        confidence_scores[build_sys] = (
                            confidence_scores.get(build_sys, 0) + 0.3
                        )

        if not detected:
            for build_sys in ["go", "cargo"]:
                for root, dirs, files in os.walk(repo_path):
                    if ".git" in root:
                        continue
                    if "go.mod" in files and build_sys == "go":
                        module_dir = root.replace(repo_path, "").lstrip("/")
                        detected.append(("go", f"{module_dir}/go.mod"))
                        confidence_scores["go"] = 0.90
                        logger.info(
                            f"Found Go module in subdirectory: {module_dir}/go.mod"
                        )
                        break
                    if "Cargo.toml" in files and build_sys == "cargo":
                        module_dir = root.replace(repo_path, "").lstrip("/")
                        detected.append(("cargo", f"{module_dir}/Cargo.toml"))
                        confidence_scores["cargo"] = 0.90
                        logger.info(
                            f"Found Cargo module in subdirectory: {module_dir}/Cargo.toml"
                        )
                        break
                if detected:
                    break

        if not detected:
            return BuildSystemInfo(
                type="unknown",
                confidence=0.0,
                primary_file="",
                additional_files=[],
            )

        best_system = max(confidence_scores.items(), key=lambda x: x[1])

        primary_file = [f for sys, f in detected if sys == best_system[0]][0]
        additional = [
            f for sys, f in detected if sys == best_system[0] and f != primary_file
        ]

        module_dir = ""
        if best_system[0] in ["go", "cargo"] and "/" in primary_file:
            module_dir = primary_file.rsplit("/", 1)[0]

        return BuildSystemInfo(
            type=best_system[0],
            confidence=min(best_system[1], 1.0),
            primary_file=primary_file,
            additional_files=additional,
            module_dir=module_dir,
        )

    # ========== Dependency Detection ==========

    def extract_dependencies(self, repo_path: str, build_system: str) -> DependencyInfo:
        """
        Extract dependencies by parsing package files.
        This is a zero-cost operation.
        """
        repo_path = self._to_host_path(repo_path)
        deps = DependencyInfo()

        if build_system == "cmake":
            deps = self._extract_cmake_dependencies(repo_path)
        elif build_system == "cargo":
            deps = self._extract_cargo_dependencies(repo_path)
        elif build_system == "pip":
            deps = self._extract_python_dependencies(repo_path)
        elif build_system == "npm":
            deps = self._extract_npm_dependencies(repo_path)
        elif build_system == "go":
            deps = self._extract_go_dependencies(repo_path)
        elif build_system == "make":
            deps = self._extract_make_dependencies(repo_path)

        return deps

    def _extract_cmake_dependencies(self, repo_path: str) -> DependencyInfo:
        """Extract dependencies from CMakeLists.txt."""
        deps = DependencyInfo(install_method="apk")
        cmake_file = os.path.join(repo_path, "CMakeLists.txt")

        if not os.path.exists(cmake_file):
            return deps

        with open(cmake_file, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        # Find find_package() calls
        package_pattern = r"find_package\s*\(\s*(\w+)"
        packages = re.findall(package_pattern, content, re.IGNORECASE)

        # Map common CMake packages to system packages (Alpine/apk names)
        package_map = {
            "Threads": "musl-dev",
            "OpenSSL": "openssl-dev",
            "ZLIB": "zlib-dev",
            "PNG": "libpng-dev",
            "JPEG": "libjpeg-turbo-dev",
            "Boost": "boost-dev",
            "Qt5": "qt5-qtbase-dev",
            "Protobuf": "protobuf-dev",
        }

        for pkg in packages:
            if pkg in package_map:
                deps.system_packages.append(package_map[pkg])
            deps.libraries.append(pkg)

        # Common build tools for CMake
        deps.build_tools = ["cmake", "make", "gcc", "g++"]

        return deps

    def _extract_cargo_dependencies(self, repo_path: str) -> DependencyInfo:
        """Extract dependencies from Cargo.toml."""
        deps = DependencyInfo(install_method="cargo")
        cargo_file = os.path.join(repo_path, "Cargo.toml")

        if not os.path.exists(cargo_file):
            return deps

        try:
            import toml

            with open(cargo_file, "r") as f:
                cargo_config = toml.load(f)

            # Extract dependencies
            if "dependencies" in cargo_config:
                deps.libraries = list(cargo_config["dependencies"].keys())
        except Exception as e:
            logger.warning(f"Failed to parse Cargo.toml: {e}")

        deps.build_tools = ["cargo", "rustc"]
        return deps

    def _extract_python_dependencies(self, repo_path: str) -> DependencyInfo:
        """Extract dependencies from requirements.txt or setup.py."""
        deps = DependencyInfo(install_method="pip")

        # Check requirements.txt
        req_file = os.path.join(repo_path, "requirements.txt")
        if os.path.exists(req_file):
            with open(req_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        # Extract package name (before ==, >=, etc.)
                        pkg = re.split(r"[=<>!]", line)[0].strip()
                        deps.libraries.append(pkg)

        deps.build_tools = ["python3", "pip"]
        return deps

    def _extract_npm_dependencies(self, repo_path: str) -> DependencyInfo:
        """Extract dependencies from package.json."""
        deps = DependencyInfo(install_method="npm")
        package_file = os.path.join(repo_path, "package.json")

        if not os.path.exists(package_file):
            return deps

        try:
            with open(package_file, "r") as f:
                package_config = json.load(f)

            if "dependencies" in package_config:
                deps.libraries.extend(package_config["dependencies"].keys())
            if "devDependencies" in package_config:
                deps.libraries.extend(package_config["devDependencies"].keys())
        except Exception as e:
            logger.warning(f"Failed to parse package.json: {e}")

        deps.build_tools = ["node", "npm"]
        return deps

    def _extract_go_dependencies(self, repo_path: str) -> DependencyInfo:
        """Extract dependencies from go.mod."""
        deps = DependencyInfo(install_method="go")
        go_mod = os.path.join(repo_path, "go.mod")

        if not os.path.exists(go_mod):
            return deps

        with open(go_mod, "r") as f:
            content = f.read()

        require_pattern = r"require\s+([^\s]+)"
        deps.libraries = re.findall(require_pattern, content)

        deps.build_tools = ["go"]
        return deps

    def find_go_main_package(self, repo_path: str) -> Dict[str, str]:
        """
        Find the Go main package location.
        Returns info about where the main package is located.
        """
        result = {
            "has_main": False,
            "main_path": "",
            "module_dir": "",
            "build_command": "go build .",
        }

        repo_path = self._to_host_path(repo_path)

        go_mod_files = []
        for root, dirs, files in os.walk(repo_path):
            if ".git" in root:
                continue
            if "go.mod" in files:
                module_dir = root.replace(repo_path, "").lstrip("/")
                go_mod_files.append((root, module_dir))

        if not go_mod_files:
            return result

        go_mod_path, module_dir = go_mod_files[0]
        result["module_dir"] = module_dir

        main_files = []
        for root, dirs, files in os.walk(go_mod_path):
            if ".git" in root or "vendor" in root:
                continue
            for f in files:
                if f.endswith(".go"):
                    filepath = os.path.join(root, f)
                    try:
                        with open(
                            filepath, "r", encoding="utf-8", errors="ignore"
                        ) as file:
                            content = file.read()
                            if "package main" in content:
                                rel_path = root.replace(go_mod_path, "").lstrip("/")
                                main_files.append(
                                    {
                                        "file": f,
                                        "dir": rel_path,
                                        "full_path": os.path.join(root, f),
                                    }
                                )
                    except:
                        pass

        if main_files:
            result["has_main"] = True
            main = main_files[0]

            if main["dir"]:
                result["main_path"] = main["dir"]
                result["build_command"] = f"go build ./{main['dir']}"
            else:
                result["main_path"] = "."
                result["build_command"] = "go build ."

        return result

    def _extract_make_dependencies(self, repo_path: str) -> DependencyInfo:
        """Extract dependencies from Makefile (basic parsing)."""
        deps = DependencyInfo(install_method="apk")
        makefile = os.path.join(repo_path, "Makefile")

        if not os.path.exists(makefile):
            return deps

        with open(makefile, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        # Look for common library flags
        lib_pattern = r"-l(\w+)"
        libs = re.findall(lib_pattern, content)

        # Map to system packages (common ones for Alpine/apk)
        lib_map = {
            "ssl": "openssl-dev",
            "crypto": "openssl-dev",
            "z": "zlib-dev",
            "pthread": "musl-dev",
            "dl": "musl-dev",
            "m": "musl-dev",
        }

        for lib in libs:
            if lib in lib_map:
                deps.system_packages.append(lib_map[lib])
            deps.libraries.append(lib)

        deps.build_tools = ["make", "gcc", "g++"]
        return deps

    # ========== Architecture-Specific Code Detection ==========

    def find_architecture_specific_code(self, repo_path: str) -> List[ArchSpecificCode]:
        """
        Search for architecture-specific code patterns.
        This is a zero-cost operation using grep.
        """
        repo_path = self._to_host_path(repo_path)
        arch_specific = []

        # Patterns to search for
        patterns = {
            "x86": [r"__x86_64__", r"__amd64__", r"__i386__", r"_M_X64", r"_M_IX86"],
            "x86_simd": [
                r"__SSE\d?__",
                r"__AVX\d?__",
                r"__FMA__",
                r"_mm\w+",
                r"__builtin_ia32",
            ],
            "arm": [r"__ARM__", r"__aarch64__", r"__arm__", r"_M_ARM"],
            "arm_simd": [r"__ARM_NEON", r"vld\d", r"vst\d", r"vmul", r"vadd"],
            "inline_asm": [r"__asm__", r"asm\s*\(", r"__asm\s+volatile"],
        }

        # Use container path for grep
        target_path = self._to_container_path(repo_path)

        for arch_type, pattern_list in patterns.items():
            for pattern in pattern_list:
                # Use grep for fast searching
                cmd = (
                    f"grep -rn -E '{pattern}' {target_path} "
                    f"--include='*.c' --include='*.cpp' --include='*.h' --include='*.hpp' "
                    f"2>/dev/null | head -n 20"
                )

                result = execute_command(cmd, use_docker=True)

                if result.success and result.stdout.strip():
                    for line in result.stdout.strip().split("\n"):
                        if ":" in line:
                            parts = line.split(":", 2)
                            if len(parts) >= 3:
                                file_path = parts[0]
                                line_num = parts[1]
                                code = parts[2][:100]  # Truncate long lines

                                # Determine severity
                                severity = "medium"
                                if arch_type in ["x86_simd", "arm_simd", "inline_asm"]:
                                    severity = "high"

                                arch_specific.append(
                                    ArchSpecificCode(
                                        file=file_path,
                                        line=int(line_num) if line_num.isdigit() else 0,
                                        code_snippet=code.strip(),
                                        arch_type=arch_type,
                                        severity=severity,
                                        suggested_fix=self._suggest_fix_for_arch_code(
                                            arch_type
                                        ),
                                    )
                                )

        return arch_specific

    def _suggest_fix_for_arch_code(self, arch_type: str) -> str:
        """Suggest fixes for architecture-specific code."""
        suggestions = {
            "x86": "Add RISC-V conditional compilation (#ifdef __riscv)",
            "x86_simd": "Use RVV (RISC-V Vector) extension or scalar fallback",
            "arm": "Add RISC-V conditional compilation (#ifdef __riscv)",
            "arm_simd": "Use RVV (RISC-V Vector) extension or scalar fallback",
            "inline_asm": "Rewrite assembly in C or add RISC-V assembly variant",
        }
        return suggestions.get(arch_type, "Review and port to RISC-V")

    # ========== File Operations ==========

    def get_file_tree(self, repo_path: str, max_depth: int = 3) -> str:
        """Get a tree view of the repository."""
        target_path = self._to_container_path(repo_path)

        cmd = f"tree -L {max_depth} -I '.git|node_modules|__pycache__|.venv' {target_path}"
        result = execute_command(cmd, use_docker=True)

        if result.success:
            return result.stdout
        else:
            cmd = f"find {target_path} -maxdepth {max_depth} -type f | head -n 100"
            result = execute_command(cmd, use_docker=True)
            return result.stdout

    def get_optimized_tree(self, repo_path: str) -> str:
        """
        Get an optimized, token-efficient tree structure for agent context.
        This provides essential structural information without full verbose output.

        Returns a compact representation showing:
        - Root level files (especially build configs, docs)
        - Directory structure (2 levels deep)
        - File counts by category
        - Key files highlighted
        """
        target_path = self._to_container_path(repo_path)

        sections = []

        sections.append("## Repository Structure Overview\n")

        ls_cmd = f"ls -la {target_path} 2>/dev/null"
        ls_result = execute_command(ls_cmd, use_docker=True)

        find_cmd = f"find {target_path} -maxdepth 2 -type d 2>/dev/null | grep -v '.git' | sort"
        find_result = execute_command(find_cmd, use_docker=True)

        if ls_result.success and ls_result.stdout.strip():
            sections.append("```\n" + ls_result.stdout.strip() + "\n```\n")

        if find_result.success and find_result.stdout.strip():
            dirs = find_result.stdout.strip().split("\n")[:20]
            if len(dirs) > 1:
                sections.append("\nSubdirectories (depth 2):\n")
                for d in dirs[:15]:
                    rel = d.replace(target_path, "").lstrip("/") or "."
                    if rel != ".":
                        sections.append(f"  {rel}/\n")

        sections.append("\n\n## Key Files Detected\n")

        key_patterns = {
            "Build Config": [
                "CMakeLists.txt",
                "Makefile",
                "Cargo.toml",
                "go.mod",
                "package.json",
                "setup.py",
                "pyproject.toml",
                "meson.build",
                "configure.ac",
                "BUILD",
                "BUILD.bazel",
                "pom.xml",
                "build.gradle",
            ],
            "Documentation": ["README*", "INSTALL*", "BUILDING*", "CONTRIBUTING*"],
            "Config": [".env*", "config.*", "*.yaml", "*.yml", "*.json"],
        }

        for category, patterns in key_patterns.items():
            found_files = []
            for pattern in patterns:
                cmd = f"find {target_path} -maxdepth 2 -name '{pattern}' -type f 2>/dev/null | head -n 5"
                result = execute_command(cmd, use_docker=True)
                if result.success and result.stdout.strip():
                    for f in result.stdout.strip().split("\n"):
                        rel_path = f.replace(target_path, "").lstrip("/")
                        if rel_path and len(found_files) < 8:
                            found_files.append(rel_path)

            if found_files:
                sections.append(f"- **{category}**: {', '.join(found_files[:8])}\n")

        sections.append("\n## Directory Stats\n")

        stats_cmd = f"find {target_path} -maxdepth 1 -type d ! -path {target_path} 2>/dev/null | wc -l"
        stats_result = execute_command(stats_cmd, use_docker=True)
        if stats_result.success:
            sections.append(f"- Root subdirectories: {stats_result.stdout.strip()}\n")

        src_cmd = f"find {target_path} -maxdepth 3 -type f -name '*.c' -o -name '*.cpp' -o -name '*.h' 2>/dev/null | wc -l"
        src_result = execute_command(src_cmd, use_docker=True)
        if src_result.success and src_result.stdout.strip() != "0":
            sections.append(f"- C/C++ source files: {src_result.stdout.strip()}\n")

        go_cmd = (
            f"find {target_path} -maxdepth 3 -type f -name '*.go' 2>/dev/null | wc -l"
        )
        go_result = execute_command(go_cmd, use_docker=True)
        if go_result.success and go_result.stdout.strip() != "0":
            sections.append(f"- Go source files: {go_result.stdout.strip()}\n")

        rs_cmd = (
            f"find {target_path} -maxdepth 3 -type f -name '*.rs' 2>/dev/null | wc -l"
        )
        rs_result = execute_command(rs_cmd, use_docker=True)
        if rs_result.success and rs_result.stdout.strip() != "0":
            sections.append(f"- Rust source files: {rs_result.stdout.strip()}\n")

        py_cmd = (
            f"find {target_path} -maxdepth 3 -type f -name '*.py' 2>/dev/null | wc -l"
        )
        py_result = execute_command(py_cmd, use_docker=True)
        if py_result.success and py_result.stdout.strip() != "0":
            sections.append(f"- Python files: {py_result.stdout.strip()}\n")

        return "\n".join(sections)

    def read_file(self, filepath: str, max_lines: int = 1000) -> str:
        """Read file content with line limit."""
        filepath = self._to_host_path(filepath)
        if not os.path.exists(filepath):
            return f"File not found: {filepath}"

        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                lines = []
                for i, line in enumerate(f):
                    if i >= max_lines:
                        lines.append(f"\n... (truncated after {max_lines} lines)")
                        break
                    lines.append(line)
                return "".join(lines)
        except Exception as e:
            return f"Error reading file: {e}"

    def search_files(
        self, repo_path: str, pattern: str, file_types: List[str] = None
    ) -> List[str]:
        """Search for files matching a pattern."""
        target_path = self._to_container_path(repo_path)

        if file_types:
            type_args = " ".join([f"-name '*.{ext}'" for ext in file_types])
            cmd = f"find {target_path} \\( {type_args} \\) -type f"
        else:
            cmd = f"find {target_path} -type f"

        result = execute_command(cmd, use_docker=True)

        if result.success:
            files = result.stdout.strip().split("\n")
            # Filter by pattern
            if pattern:
                files = [f for f in files if re.search(pattern, f, re.IGNORECASE)]
            return files

        return []

    # ========== Documentation Search ==========

    def find_documentation(self, repo_path: str) -> List[str]:
        """Find common documentation files."""
        target_path = self._to_container_path(repo_path)
        doc_patterns = [
            "README*",
            "INSTALL*",
            "BUILDING*",
            "BUILD*",
            "CONTRIBUTING*",
            "docs/*.md",
            "doc/*.md",
            "*.md",
        ]

        found_docs = []
        for pattern in doc_patterns:
            cmd = f"find {target_path} -maxdepth 2 -iname '{pattern}' -type f"
            result = execute_command(cmd, use_docker=True)

            if result.success and result.stdout.strip():
                found_docs.extend(result.stdout.strip().split("\n"))

        # Remove duplicates and sort by likely importance
        unique_docs = list(set(found_docs))

        # Prioritize certain files
        priority = ["README", "INSTALL", "BUILDING", "BUILD"]
        sorted_docs = []

        for p in priority:
            sorted_docs.extend([d for d in unique_docs if p in d.upper()])

        # Add remaining docs
        sorted_docs.extend([d for d in unique_docs if d not in sorted_docs])

        return sorted_docs[:10]  # Limit to top 10

    def get_system_info(self, tools: Optional[List[str]] = None) -> Dict[str, str]:
        """Get information about the system environment inside the CONTAINER."""
        info = {}
        if tools is None:
            tools = [
                "gcc",
                "g++",
                "cmake",
                "make",
                "ninja",
                "meson",
                "automake",
                "autoconf",
                "python3",
                "go",
                "rustc",
                "cargo",
            ]

        for tool in tools:
            # Check availability INSIDE the container
            res = execute_command(f"which {tool}", use_docker=True)
            if res.success:
                info[tool] = "Available in PATH"
            else:
                info[tool] = "Not installed"

        # Architecture check
        arch_res = execute_command("uname -m", use_docker=True)
        info["architecture"] = (
            arch_res.stdout.strip() if arch_res.success else "unknown"
        )

        return info

    # ========== Helper Methods ==========

    def detect_arch_specific_build_files(self, repo_path: str) -> Dict[str, Any]:
        """
        Detect architecture-specific build files in the repository.
        Returns information about existing arch files and whether RISC-V support exists.
        """
        target_path = self._to_container_path(repo_path)
        result = {
            "has_arch_specific": False,
            "archs_found": [],
            "riscv_exists": False,
            "arch_files": {},
            "suggested_riscv_files": [],
        }

        arch_patterns = [
            ("x64", r"(cmpl_gcc_|var_gcc_|makefile\.|build_)(x64|x86_64|amd64)"),
            ("arm64", r"(cmpl_gcc_|var_gcc_|makefile\.|build_)(arm64|aarch64)"),
            ("x86", r"(cmpl_gcc_|var_gcc_|makefile\.|build_)(x86|i386|i686)"),
            ("riscv", r"(cmpl_gcc_|var_gcc_|makefile\.|build_)(riscv|riscv64|rv64)"),
        ]

        for arch, pattern in arch_patterns:
            cmd = f"find {target_path} -type f -name '*.mak' -o -name '*.mk' 2>/dev/null | xargs grep -l '{arch}' 2>/dev/null | head -20"
            find_result = execute_command(cmd, use_docker=True)

            if find_result.success and find_result.stdout.strip():
                files = find_result.stdout.strip().split("\n")
                result["has_arch_specific"] = True
                if arch not in result["archs_found"]:
                    result["archs_found"].append(arch)
                result["arch_files"][arch] = [f for f in files if f]

        if "riscv" in result["archs_found"]:
            result["riscv_exists"] = True
        else:
            for arch in ["x64", "arm64"]:
                if arch in result["arch_files"]:
                    for src_file in result["arch_files"][arch][:3]:
                        riscv_file = src_file.replace(arch, "riscv64")
                        riscv_file = re.sub(r"x86_64|amd64", "riscv64", riscv_file)
                        result["suggested_riscv_files"].append(
                            {
                                "source": src_file,
                                "target": riscv_file,
                            }
                        )

        return result

    def clear_cache(self):
        """Clear all cached data."""
        import shutil

        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


def quick_analysis(repo_path: str) -> Dict[str, Any]:
    """
    Perform quick repository analysis using only scripted operations.
    This provides initial context without any LLM calls.
    """
    ops = ScriptedOperations()

    analysis = {
        "repo_info": ops.get_repository_info(repo_path),
        "build_system": ops.detect_build_system(repo_path),
        "file_tree": ops.get_file_tree(repo_path, max_depth=2),
        "optimized_tree": ops.get_optimized_tree(repo_path),
        "documentation": ops.find_documentation(repo_path),
        "arch_build_files": ops.detect_arch_specific_build_files(repo_path),
    }

    if analysis["build_system"].type != "unknown":
        analysis["dependencies"] = ops.extract_dependencies(
            repo_path, analysis["build_system"].type
        )

        if analysis["build_system"].type == "go":
            analysis["go_main_info"] = ops.find_go_main_package(repo_path)

    analysis["arch_specific_code"] = ops.find_architecture_specific_code(repo_path)

    return analysis
