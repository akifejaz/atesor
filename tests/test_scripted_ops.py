import unittest
from unittest.mock import patch, MagicMock
import os
import json
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

class TestPathTranslation(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.tmpdir = tempfile.mkdtemp()
        self.ops = ScriptedOperations(workspace_root=self.tmpdir)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    @patch('os.path.exists')
    def test_to_host_path_translates_when_workspace_missing(self, mock_exists):
        mock_exists.return_value = False
        result = self.ops._to_host_path("/workspace/repos/myrepo")
        self.assertEqual(result, os.path.join(self.tmpdir, "repos/myrepo"))

    @patch('os.path.exists')
    def test_to_host_path_passthrough_when_workspace_exists(self, mock_exists):
        mock_exists.return_value = True
        result = self.ops._to_host_path("/workspace/repos/myrepo")
        self.assertEqual(result, "/workspace/repos/myrepo")

    def test_to_host_path_non_workspace_path(self):
        result = self.ops._to_host_path("/other/path/repo")
        self.assertEqual(result, "/other/path/repo")

    def test_to_container_path_translates_host_path(self):
        result = self.ops._to_container_path(os.path.join(self.tmpdir, "repos/myrepo"))
        self.assertEqual(result, "/workspace/repos/myrepo")

    def test_to_container_path_passthrough_for_container_path(self):
        result = self.ops._to_container_path("/workspace/repos/myrepo")
        self.assertEqual(result, "/workspace/repos/myrepo")

    def test_to_container_path_passthrough_for_unrelated_path(self):
        result = self.ops._to_container_path("/some/other/path")
        self.assertEqual(result, "/some/other/path")


class TestDetectBuildSystemExtended(unittest.TestCase):
    def setUp(self):
        self.workspace = "/tmp/test_workspace"
        self.ops = ScriptedOperations(workspace_root=self.workspace)

    @patch('os.path.exists')
    def test_detect_go(self, mock_exists):
        mock_exists.side_effect = lambda p: "go.mod" in p
        info = self.ops.detect_build_system("/workspace/repo")
        self.assertEqual(info.type, "go")
        self.assertEqual(info.primary_file, "go.mod")
        self.assertGreater(info.confidence, 0.9)

    @patch('os.path.exists')
    def test_detect_pip(self, mock_exists):
        mock_exists.side_effect = lambda p: "setup.py" in p
        info = self.ops.detect_build_system("/workspace/repo")
        self.assertEqual(info.type, "pip")
        self.assertEqual(info.primary_file, "setup.py")

    @patch('os.path.exists')
    def test_detect_npm(self, mock_exists):
        mock_exists.side_effect = lambda p: "package.json" in p
        info = self.ops.detect_build_system("/workspace/repo")
        self.assertEqual(info.type, "npm")
        self.assertEqual(info.primary_file, "package.json")

    @patch('os.path.exists')
    def test_detect_make(self, mock_exists):
        mock_exists.side_effect = lambda p: "Makefile" in p
        info = self.ops.detect_build_system("/workspace/repo")
        self.assertEqual(info.type, "make")
        self.assertEqual(info.primary_file, "Makefile")

    @patch('os.path.exists')
    def test_detect_meson(self, mock_exists):
        mock_exists.side_effect = lambda p: "meson.build" in p
        info = self.ops.detect_build_system("/workspace/repo")
        self.assertEqual(info.type, "meson")
        self.assertEqual(info.primary_file, "meson.build")

    @patch('os.path.exists')
    def test_detect_autotools(self, mock_exists):
        mock_exists.side_effect = lambda p: "configure.ac" in p
        info = self.ops.detect_build_system("/workspace/repo")
        self.assertEqual(info.type, "autotools")
        self.assertEqual(info.primary_file, "configure.ac")
        self.assertGreater(info.confidence, 0.9)

    @patch('os.path.exists')
    def test_detect_maven(self, mock_exists):
        mock_exists.side_effect = lambda p: "pom.xml" in p
        info = self.ops.detect_build_system("/workspace/repo")
        self.assertEqual(info.type, "maven")
        self.assertEqual(info.primary_file, "pom.xml")

    @patch('os.walk')
    @patch('os.path.exists')
    def test_detect_unknown(self, mock_exists, mock_walk):
        mock_exists.return_value = False
        mock_walk.return_value = []
        info = self.ops.detect_build_system("/workspace/repo")
        self.assertEqual(info.type, "unknown")
        self.assertEqual(info.confidence, 0.0)
        self.assertEqual(info.primary_file, "")

    @patch('os.path.exists')
    def test_detect_multiple_cmake_and_make(self, mock_exists):
        def side_effect(path):
            return "CMakeLists.txt" in path or "Makefile" in path
        mock_exists.side_effect = side_effect
        info = self.ops.detect_build_system("/workspace/repo")
        self.assertEqual(info.type, "cmake")
        self.assertGreater(info.confidence, 0.9)


class TestExtractDependencies(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.workspace = "/tmp/test_workspace"
        self.ops = ScriptedOperations(workspace_root=self.workspace)
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_cmake_dependencies(self):
        cmake_content = """
cmake_minimum_required(VERSION 3.10)
project(MyProject)
find_package(OpenSSL REQUIRED)
find_package(ZLIB REQUIRED)
find_package(Threads REQUIRED)
"""
        with open(os.path.join(self.tmpdir, "CMakeLists.txt"), "w") as f:
            f.write(cmake_content)
        deps = self.ops.extract_dependencies(self.tmpdir, "cmake")
        self.assertIn("OpenSSL", deps.libraries)
        self.assertIn("ZLIB", deps.libraries)
        self.assertIn("Threads", deps.libraries)
        self.assertIn("openssl-dev", deps.system_packages)
        self.assertIn("zlib-dev", deps.system_packages)
        self.assertIn("musl-dev", deps.system_packages)
        self.assertIn("cmake", deps.build_tools)
        self.assertIn("make", deps.build_tools)

    def test_cargo_dependencies(self):
        cargo_content = """[package]
name = "myapp"
version = "0.1.0"

[dependencies]
serde = "1.0"
tokio = { version = "1", features = ["full"] }
"""
        with open(os.path.join(self.tmpdir, "Cargo.toml"), "w") as f:
            f.write(cargo_content)
        deps = self.ops.extract_dependencies(self.tmpdir, "cargo")
        self.assertIn("serde", deps.libraries)
        self.assertIn("tokio", deps.libraries)
        self.assertIn("cargo", deps.build_tools)
        self.assertIn("rustc", deps.build_tools)
        self.assertEqual(deps.install_method, "cargo")

    def test_python_dependencies(self):
        req_content = """requests>=2.28.0
flask==2.3.0
# A comment
numpy
pandas>=1.5
"""
        with open(os.path.join(self.tmpdir, "requirements.txt"), "w") as f:
            f.write(req_content)
        deps = self.ops.extract_dependencies(self.tmpdir, "pip")
        self.assertIn("requests", deps.libraries)
        self.assertIn("flask", deps.libraries)
        self.assertIn("numpy", deps.libraries)
        self.assertIn("pandas", deps.libraries)
        self.assertNotIn("# A comment", deps.libraries)
        self.assertEqual(deps.install_method, "pip")

    def test_npm_dependencies(self):
        package_content = json.dumps({
            "name": "myapp",
            "dependencies": {"express": "^4.18.0", "lodash": "^4.17.21"},
            "devDependencies": {"jest": "^29.0.0"},
        })
        with open(os.path.join(self.tmpdir, "package.json"), "w") as f:
            f.write(package_content)
        deps = self.ops.extract_dependencies(self.tmpdir, "npm")
        self.assertIn("express", deps.libraries)
        self.assertIn("lodash", deps.libraries)
        self.assertIn("jest", deps.libraries)
        self.assertIn("node", deps.build_tools)
        self.assertIn("npm", deps.build_tools)
        self.assertEqual(deps.install_method, "npm")

    def test_go_dependencies(self):
        go_mod_content = """module github.com/example/myapp

go 1.21

require github.com/gin-gonic/gin v1.9.0
require github.com/stretchr/testify v1.8.0
"""
        with open(os.path.join(self.tmpdir, "go.mod"), "w") as f:
            f.write(go_mod_content)
        deps = self.ops.extract_dependencies(self.tmpdir, "go")
        self.assertIn("github.com/gin-gonic/gin", deps.libraries)
        self.assertIn("github.com/stretchr/testify", deps.libraries)
        self.assertIn("go", deps.build_tools)
        self.assertEqual(deps.install_method, "go")

    def test_make_dependencies(self):
        makefile_content = """
CC=gcc
CFLAGS=-Wall
LDFLAGS=-lssl -lcrypto -lz -lpthread

all:
\t$(CC) $(CFLAGS) -o myapp main.c $(LDFLAGS)
"""
        with open(os.path.join(self.tmpdir, "Makefile"), "w") as f:
            f.write(makefile_content)
        deps = self.ops.extract_dependencies(self.tmpdir, "make")
        self.assertIn("ssl", deps.libraries)
        self.assertIn("crypto", deps.libraries)
        self.assertIn("z", deps.libraries)
        self.assertIn("pthread", deps.libraries)
        self.assertIn("openssl-dev", deps.system_packages)
        self.assertIn("zlib-dev", deps.system_packages)
        self.assertIn("musl-dev", deps.system_packages)
        self.assertEqual(deps.install_method, "apk")

    def test_unknown_build_system(self):
        deps = self.ops.extract_dependencies(self.tmpdir, "unknown_system")
        self.assertEqual(deps.libraries, [])
        self.assertEqual(deps.system_packages, [])
        self.assertEqual(deps.build_tools, [])


class TestSuggestFixForArchCode(unittest.TestCase):
    def setUp(self):
        self.workspace = "/tmp/test_workspace"
        self.ops = ScriptedOperations(workspace_root=self.workspace)

    def test_x86_suggestion(self):
        result = self.ops._suggest_fix_for_arch_code("x86")
        self.assertIn("RISC-V", result)
        self.assertIn("__riscv", result)

    def test_x86_simd_suggestion(self):
        result = self.ops._suggest_fix_for_arch_code("x86_simd")
        self.assertIn("RVV", result)

    def test_arm_suggestion(self):
        result = self.ops._suggest_fix_for_arch_code("arm")
        self.assertIn("RISC-V", result)
        self.assertIn("__riscv", result)

    def test_arm_simd_suggestion(self):
        result = self.ops._suggest_fix_for_arch_code("arm_simd")
        self.assertIn("RVV", result)

    def test_inline_asm_suggestion(self):
        result = self.ops._suggest_fix_for_arch_code("inline_asm")
        self.assertIn("assembly", result.lower())

    def test_unknown_arch_suggestion(self):
        result = self.ops._suggest_fix_for_arch_code("some_unknown_arch")
        self.assertIn("RISC-V", result)
        self.assertEqual(result, "Review and port to RISC-V")


class TestFindGoMainPackage(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.workspace = "/tmp/test_workspace"
        self.ops = ScriptedOperations(workspace_root=self.workspace)
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_find_main_package(self):
        with open(os.path.join(self.tmpdir, "go.mod"), "w") as f:
            f.write("module github.com/example/myapp\n\ngo 1.21\n")
        with open(os.path.join(self.tmpdir, "main.go"), "w") as f:
            f.write("package main\n\nfunc main() {}\n")
        result = self.ops.find_go_main_package(self.tmpdir)
        self.assertTrue(result["has_main"])
        self.assertEqual(result["main_path"], ".")
        self.assertEqual(result["build_command"], "go build .")

    def test_no_go_mod(self):
        result = self.ops.find_go_main_package(self.tmpdir)
        self.assertFalse(result["has_main"])
        self.assertEqual(result["main_path"], "")

    def test_go_mod_but_no_main_package(self):
        with open(os.path.join(self.tmpdir, "go.mod"), "w") as f:
            f.write("module github.com/example/mylib\n\ngo 1.21\n")
        os.makedirs(os.path.join(self.tmpdir, "pkg"), exist_ok=True)
        with open(os.path.join(self.tmpdir, "pkg", "util.go"), "w") as f:
            f.write("package util\n\nfunc Helper() {}\n")
        result = self.ops.find_go_main_package(self.tmpdir)
        self.assertFalse(result["has_main"])
        self.assertEqual(result["main_path"], "")

    def test_main_in_subdirectory(self):
        with open(os.path.join(self.tmpdir, "go.mod"), "w") as f:
            f.write("module github.com/example/myapp\n\ngo 1.21\n")
        cmd_dir = os.path.join(self.tmpdir, "cmd", "server")
        os.makedirs(cmd_dir, exist_ok=True)
        with open(os.path.join(cmd_dir, "main.go"), "w") as f:
            f.write("package main\n\nfunc main() {}\n")
        result = self.ops.find_go_main_package(self.tmpdir)
        self.assertTrue(result["has_main"])
        self.assertIn("cmd/server", result["main_path"])
        self.assertIn("cmd/server", result["build_command"])


class TestExtractMakeDependencies(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.workspace = "/tmp/test_workspace"
        self.ops = ScriptedOperations(workspace_root=self.workspace)
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_make_lib_flags(self):
        makefile_content = """
LIBS=-lssl -lcrypto -lz -lpthread
CFLAGS=-Wall -O2

all: myapp

myapp: main.o
\t$(CC) -o $@ $^ $(LIBS)
"""
        with open(os.path.join(self.tmpdir, "Makefile"), "w") as f:
            f.write(makefile_content)
        deps = self.ops._extract_make_dependencies(self.tmpdir)
        self.assertIn("ssl", deps.libraries)
        self.assertIn("crypto", deps.libraries)
        self.assertIn("z", deps.libraries)
        self.assertIn("pthread", deps.libraries)
        self.assertIn("openssl-dev", deps.system_packages)
        self.assertIn("zlib-dev", deps.system_packages)
        self.assertIn("musl-dev", deps.system_packages)
        self.assertEqual(deps.install_method, "apk")
        self.assertIn("make", deps.build_tools)
        self.assertIn("gcc", deps.build_tools)

    def test_no_makefile(self):
        deps = self.ops._extract_make_dependencies(self.tmpdir)
        self.assertEqual(deps.libraries, [])
        self.assertEqual(deps.system_packages, [])


if __name__ == "__main__":
    unittest.main()
