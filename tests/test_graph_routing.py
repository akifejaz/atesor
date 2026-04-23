import unittest

from langgraph.graph import END

from src.graph import (
    route_next,
    extract_content,
    extract_json_block,
    _build_command_error_message,
    validate_build_plan,
    validate_fix_command,
    validate_fixer_response,
    create_default_plan,
    create_fallback_build_plan,
    predict_build_issues,
)
from src.state import (
    BuildStatus,
    create_initial_state,
    CommandResult,
    BuildPlan,
    BuildPhase,
    BuildSystemInfo,
    ArchSpecificCode,
    TaskPlan,
    TaskPhase,
    AgentRole,
)


class TestGraphRouting(unittest.TestCase):
    def test_failed_initialization_routes_to_escalation(self):
        state = create_initial_state("https://github.com/example/repo")
        state.current_phase = "initialization"
        state.build_status = BuildStatus.FAILED

        self.assertEqual(route_next(state), "escalate_node")


# ============================================================================
# TestExtractContent
# ============================================================================


class TestExtractContent(unittest.TestCase):
    def test_string_input(self):
        self.assertEqual(extract_content("hello"), "hello")

    def test_list_of_strings(self):
        self.assertEqual(extract_content(["a", "b", "c"]), "a\nb\nc")

    def test_list_of_dicts_with_text_key(self):
        result = extract_content([{"text": "line1"}, {"text": "line2"}])
        self.assertEqual(result, "line1\nline2")

    def test_list_of_mixed(self):
        result = extract_content(["plain", {"text": "rich"}, 42])
        self.assertEqual(result, "plain\nrich\n42")

    def test_non_string_non_list(self):
        self.assertEqual(extract_content(123), "123")
        self.assertEqual(extract_content(None), "None")


# ============================================================================
# TestExtractJsonBlock
# ============================================================================


class TestExtractJsonBlock(unittest.TestCase):
    def test_text_with_json_block(self):
        text = 'Here is JSON: {"key": "value"} done'
        self.assertEqual(extract_json_block(text), '{"key": "value"}')

    def test_no_json(self):
        text = "no json here"
        self.assertEqual(extract_json_block(text), text)

    def test_only_braces(self):
        self.assertEqual(extract_json_block("{}"), "{}")

    def test_multiple_json_blocks_returns_outer(self):
        text = 'a {"inner": {"nested": true}} b'
        self.assertEqual(extract_json_block(text), '{"inner": {"nested": true}}')

    def test_text_before_and_after_json(self):
        text = 'prefix {"a": 1} suffix'
        self.assertEqual(extract_json_block(text), '{"a": 1}')

    def test_no_closing_brace(self):
        text = "only opening {"
        self.assertEqual(extract_json_block(text), text)

    def test_no_opening_brace(self):
        text = "only closing }"
        self.assertEqual(extract_json_block(text), text)


# ============================================================================
# TestBuildCommandErrorMessage
# ============================================================================


class TestBuildCommandErrorMessage(unittest.TestCase):
    def _make_result(self, exit_code=1, stderr=None, stdout=None):
        from datetime import datetime

        return CommandResult(
            command="test",
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            duration_seconds=0.1,
            timestamp=datetime.now(),
        )

    def test_stderr_only(self):
        result = self._make_result(stderr="error msg", stdout="")
        msg = _build_command_error_message(result, "Build failed")
        self.assertIn("error msg", msg)
        self.assertIn("exit 1", msg)
        self.assertIn("Build failed", msg)

    def test_stdout_only(self):
        result = self._make_result(stderr="", stdout="output msg")
        msg = _build_command_error_message(result, "Build failed")
        self.assertIn("output msg", msg)

    def test_neither_stderr_nor_stdout(self):
        result = self._make_result(stderr="", stdout="")
        msg = _build_command_error_message(result, "Build failed")
        self.assertIn("No stderr/stdout output captured.", msg)

    def test_both_stderr_preferred(self):
        result = self._make_result(stderr="err", stdout="out")
        msg = _build_command_error_message(result, "Build failed")
        self.assertIn("err", msg)
        self.assertNotIn("out", msg)

    def test_none_values(self):
        result = self._make_result(stderr=None, stdout=None)
        msg = _build_command_error_message(result, "fallback")
        self.assertIn("No stderr/stdout output captured.", msg)


# ============================================================================
# TestValidateBuildPlan
# ============================================================================


class TestValidateBuildPlan(unittest.TestCase):
    def _make_plan(self, commands):
        return BuildPlan(
            build_system="test",
            build_system_confidence=0.9,
            phases=[BuildPhase(1, "build", commands)],
            total_estimated_duration="5m",
        )

    def test_clean_plan_is_valid(self):
        plan = self._make_plan(["make -j4", "make install"])
        valid, msg = validate_build_plan(plan)
        self.assertTrue(valid)
        self.assertEqual(msg, "")

    def test_path_to_hallucination(self):
        plan = self._make_plan(["cd /path/to/source && make"])
        valid, msg = validate_build_plan(plan)
        self.assertFalse(valid)
        self.assertIn("/path/to/", msg)

    def test_your_username_hallucination(self):
        plan = self._make_plan(["cd /home/your_username/project && make"])
        valid, msg = validate_build_plan(plan)
        self.assertFalse(valid)
        self.assertIn("your_username", msg)

    def test_example_com_hallucination(self):
        plan = self._make_plan(["curl http://example.com/pkg.tar.gz | tar xz"])
        valid, msg = validate_build_plan(plan)
        self.assertFalse(valid)
        self.assertIn("example.com", msg)

    def test_home_path_suspicious(self):
        plan = self._make_plan(["cd /home/user/project && make"])
        valid, msg = validate_build_plan(plan)
        self.assertFalse(valid)
        self.assertIn("Suspicious absolute path", msg)

    def test_home_workspace_path_allowed(self):
        plan = self._make_plan(["cd /home/user/workspace/project && make"])
        valid, msg = validate_build_plan(plan)
        self.assertTrue(valid)

    def test_bad_cmake_pattern(self):
        plan = self._make_plan(["cd build && cmake -S . -B build .."])
        valid, msg = validate_build_plan(plan)
        self.assertFalse(valid)
        self.assertIn("cmake", msg.lower())

    def test_all_valid_commands(self):
        plan = self._make_plan([
            "apk update && apk add build-base",
            "mkdir -p build",
            "cd build && cmake ..",
            "make -j$(nproc)",
        ])
        valid, msg = validate_build_plan(plan)
        self.assertTrue(valid)


# ============================================================================
# TestValidateFixCommand
# ============================================================================


class TestValidateFixCommand(unittest.TestCase):
    def test_safe_commands(self):
        safe, reason = validate_fix_command("apk add build-base")
        self.assertTrue(safe)
        self.assertEqual(reason, "Command is safe")

    def test_touch_go_file(self):
        safe, reason = validate_fix_command("touch main.go")
        self.assertFalse(safe)
        self.assertIn("Go", reason)

    def test_touch_c_file(self):
        safe, reason = validate_fix_command("touch file.c")
        self.assertFalse(safe)
        self.assertIn("C source", reason)

    def test_rm_rf_root(self):
        safe, reason = validate_fix_command("rm -rf /")
        self.assertFalse(safe)
        self.assertIn("root", reason.lower())

    def test_git_push(self):
        safe, reason = validate_fix_command("git push origin main")
        self.assertFalse(safe)
        self.assertIn("push", reason.lower())

    def test_git_reset_hard(self):
        safe, reason = validate_fix_command("git reset --hard HEAD~1")
        self.assertFalse(safe)
        self.assertIn("hard reset", reason.lower())

    def test_empty_redirect(self):
        safe, reason = validate_fix_command("cat > ")
        self.assertFalse(safe)
        self.assertIn("redirect", reason.lower())

    def test_echo_empty_overwrite(self):
        safe, reason = validate_fix_command('echo "" > file.txt')
        self.assertFalse(safe)
        self.assertIn("empty content", reason.lower())


# ============================================================================
# TestValidateFixerResponse
# ============================================================================


class TestValidateFixerResponse(unittest.TestCase):
    def _valid_response(self):
        return {
            "strategies": [
                {
                    "id": 1,
                    "actions": [
                        {"type": "command", "command": "apk add build-base"},
                    ],
                }
            ],
            "recommended_strategy_id": 1,
            "reflection": {
                "root_cause": "missing deps",
                "this_fix_will_work_because": "installs them",
            },
        }

    def test_valid_response(self):
        valid, msg = validate_fixer_response(self._valid_response())
        self.assertTrue(valid)
        self.assertEqual(msg, "")

    def test_no_strategies(self):
        valid, msg = validate_fixer_response({"strategies": []})
        self.assertFalse(valid)
        self.assertIn("No strategies", msg)

    def test_bad_recommended_id(self):
        data = self._valid_response()
        data["recommended_strategy_id"] = 99
        valid, msg = validate_fixer_response(data)
        self.assertFalse(valid)
        self.assertIn("not found", msg)

    def test_empty_actions(self):
        data = self._valid_response()
        data["strategies"][0]["actions"] = []
        valid, msg = validate_fixer_response(data)
        self.assertFalse(valid)
        self.assertIn("no actions", msg)

    def test_create_file_with_empty_content(self):
        data = self._valid_response()
        data["strategies"][0]["actions"] = [
            {"type": "create_file", "path": "foo.c", "content": ""},
        ]
        valid, msg = validate_fixer_response(data)
        self.assertFalse(valid)
        self.assertIn("empty content", msg)

    def test_create_file_with_long_path(self):
        data = self._valid_response()
        data["strategies"][0]["actions"] = [
            {"type": "create_file", "path": "a" * 150, "content": "int main(){}"},
        ]
        valid, msg = validate_fixer_response(data)
        self.assertFalse(valid)
        self.assertIn("long filepath", msg.lower())

    def test_create_file_with_home_path(self):
        data = self._valid_response()
        data["strategies"][0]["actions"] = [
            {"type": "create_file", "path": "/home/user/foo.c", "content": "int main(){}"},
        ]
        valid, msg = validate_fixer_response(data)
        self.assertFalse(valid)
        self.assertIn("absolute path", msg.lower())

    def test_empty_command(self):
        data = self._valid_response()
        data["strategies"][0]["actions"] = [
            {"type": "command", "command": ""},
        ]
        valid, msg = validate_fixer_response(data)
        self.assertFalse(valid)
        self.assertIn("empty", msg.lower())


# ============================================================================
# TestCreateDefaultPlan
# ============================================================================


class TestCreateDefaultPlan(unittest.TestCase):
    def test_returns_task_plan_with_two_phases(self):
        plan = create_default_plan()
        self.assertIsInstance(plan, TaskPlan)
        self.assertEqual(len(plan.phases), 2)

    def test_correct_agent_roles(self):
        plan = create_default_plan()
        self.assertEqual(plan.phases[0].agent, AgentRole.SCOUT)
        self.assertEqual(plan.phases[1].agent, AgentRole.BUILDER)

    def test_phase_ids(self):
        plan = create_default_plan()
        self.assertEqual(plan.phases[0].id, 1)
        self.assertEqual(plan.phases[1].id, 2)

    def test_complexity_score(self):
        plan = create_default_plan()
        self.assertEqual(plan.complexity_score, 5)


# ============================================================================
# TestCreateFallbackBuildPlan
# ============================================================================


class TestCreateFallbackBuildPlan(unittest.TestCase):
    def _make_state(self, build_type):
        state = create_initial_state("https://github.com/example/repo")
        state.build_system_info = BuildSystemInfo(
            type=build_type,
            confidence=0.9,
            primary_file="Makefile",
        )
        return state

    def test_go_build_plan(self):
        plan = create_fallback_build_plan(self._make_state("go"))
        self.assertEqual(plan.build_system, "go")
        cmds = [c for p in plan.phases for c in p.commands]
        self.assertTrue(any("go build" in c for c in cmds))

    def test_cmake_build_plan(self):
        plan = create_fallback_build_plan(self._make_state("cmake"))
        self.assertEqual(plan.build_system, "cmake")
        cmds = [c for p in plan.phases for c in p.commands]
        self.assertTrue(any("cmake" in c for c in cmds))

    def test_make_build_plan(self):
        plan = create_fallback_build_plan(self._make_state("make"))
        self.assertEqual(plan.build_system, "make")
        cmds = [c for p in plan.phases for c in p.commands]
        self.assertTrue(any("make" in c for c in cmds))

    def test_cargo_build_plan(self):
        plan = create_fallback_build_plan(self._make_state("cargo"))
        self.assertEqual(plan.build_system, "cargo")
        cmds = [c for p in plan.phases for c in p.commands]
        self.assertTrue(any("cargo build" in c for c in cmds))

    def test_meson_build_plan(self):
        plan = create_fallback_build_plan(self._make_state("meson"))
        self.assertEqual(plan.build_system, "meson")
        cmds = [c for p in plan.phases for c in p.commands]
        self.assertTrue(any("meson" in c for c in cmds))

    def test_unknown_build_system(self):
        plan = create_fallback_build_plan(self._make_state("unknown_sys"))
        self.assertEqual(plan.build_system, "unknown_sys")
        self.assertLess(plan.build_system_confidence, 0.5)
        self.assertTrue(any("Generic fallback" in n for n in plan.notes))


# ============================================================================
# TestPredictBuildIssues
# ============================================================================


class TestPredictBuildIssues(unittest.TestCase):
    def _make_state_with_plan(self, commands, arch_code=None, missing_tools=None):
        state = create_initial_state("https://github.com/example/repo")
        state.build_plan = BuildPlan(
            build_system="test",
            build_system_confidence=0.9,
            phases=[BuildPhase(1, "build", commands)],
            total_estimated_duration="5m",
        )
        if arch_code:
            state.arch_specific_code = arch_code
        if missing_tools:
            state.context_cache["missing_tools"] = missing_tools
        return state

    def test_go_build_without_buildvcs_flag(self):
        state = self._make_state_with_plan(["go build ./..."])
        predictions = predict_build_issues(state)
        self.assertTrue(len(predictions) > 0)
        self.assertTrue(any("VCS" in p["issue"] for p in predictions))

    def test_go_build_with_buildvcs_flag_no_issue(self):
        state = self._make_state_with_plan(["go build -buildvcs=false ./..."])
        predictions = predict_build_issues(state)
        vcs_issues = [p for p in predictions if "VCS" in p["issue"]]
        self.assertEqual(len(vcs_issues), 0)

    def test_cmake_with_high_severity_arch_code(self):
        arch = [ArchSpecificCode(
            file="src/simd.c", line=10, code_snippet="asm volatile",
            arch_type="x86", severity="high",
        )]
        state = self._make_state_with_plan(["cmake .."], arch_code=arch)
        predictions = predict_build_issues(state)
        self.assertTrue(any("arch" in p["issue"].lower() for p in predictions))

    def test_missing_tools(self):
        state = self._make_state_with_plan(
            ["apk add build-base"], missing_tools=["ninja", "meson"]
        )
        predictions = predict_build_issues(state)
        self.assertTrue(any("Missing tools" in p["issue"] for p in predictions))

    def test_no_build_plan_returns_empty(self):
        state = create_initial_state("https://github.com/example/repo")
        state.build_plan = None
        predictions = predict_build_issues(state)
        self.assertEqual(predictions, [])

    def test_clean_plan_no_issues(self):
        state = self._make_state_with_plan(["make -j4"])
        predictions = predict_build_issues(state)
        self.assertEqual(predictions, [])


# ============================================================================
# TestRouteNextExtended
# ============================================================================


class TestRouteNextExtended(unittest.TestCase):
    def _route(self, phase, status=BuildStatus.PENDING):
        state = create_initial_state("https://github.com/example/repo")
        state.current_phase = phase
        state.build_status = status
        return route_next(state)

    def test_initialization_routes_to_planner(self):
        self.assertEqual(self._route("initialization"), "planner")

    def test_planned_routes_to_supervisor(self):
        self.assertEqual(self._route("planned"), "supervisor")

    def test_scout_routes_to_scout_node(self):
        self.assertEqual(self._route("scout"), "scout_node")

    def test_scouting_routes_to_supervisor(self):
        self.assertEqual(self._route("scouting"), "supervisor")

    def test_builder_routes_to_builder_node(self):
        self.assertEqual(self._route("builder"), "builder_node")

    def test_building_routes_to_supervisor(self):
        self.assertEqual(self._route("building"), "supervisor")

    def test_fixer_routes_to_fixer_node(self):
        self.assertEqual(self._route("fixer"), "fixer_node")

    def test_fixing_routes_to_supervisor(self):
        self.assertEqual(self._route("fixing"), "supervisor")

    def test_escalate_routes_to_escalate_node(self):
        self.assertEqual(self._route("escalate"), "escalate_node")

    def test_escalated_routes_to_end(self):
        self.assertEqual(self._route("escalated"), END)

    def test_finish_routes_to_finish_node(self):
        self.assertEqual(self._route("finish"), "finish_node")

    def test_finished_routes_to_end(self):
        self.assertEqual(self._route("finished"), END)

    def test_unknown_defaults_to_supervisor(self):
        self.assertEqual(self._route("something_unknown"), "supervisor")

    def test_failed_initialization_overrides_routing_map(self):
        self.assertEqual(self._route("initialization", BuildStatus.FAILED), "escalate_node")


if __name__ == "__main__":
    unittest.main()
