import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage

from src.graph import fixer_node
from src.state import BuildStatus, ErrorCategory, create_initial_state


class _FakeLLM:
    def __init__(self, content: str):
        self._content = content

    def invoke(self, messages):
        return AIMessage(content=self._content)


class TestFixerBehavior(unittest.TestCase):
    @patch("src.graph.get_model_for_role")
    @patch("src.graph.format_few_shot_examples")
    def test_fixer_fails_when_no_effective_changes(self, mock_examples, mock_model):
        mock_examples.return_value = ""
        mock_model.return_value = _FakeLLM(
            '{"strategies":[{"id":1,"description":"noop","actions":[]}],"recommended_strategy_id":1}'
        )

        state = create_initial_state("https://github.com/example/repo")
        state.build_status = BuildStatus.FAILED
        state.last_error = "compile failed"
        state.last_error_category = ErrorCategory.COMPILATION

        updated = fixer_node(state)

        self.assertEqual(updated.build_status, BuildStatus.FAILED)
        self.assertTrue(any("no effective changes" in n.lower() for n in updated.feedback_notes))


if __name__ == "__main__":
    unittest.main()
