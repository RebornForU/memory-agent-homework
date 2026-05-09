import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.memory_manager import MemoryManager


LONG_HISTORY = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain quantum computing in detail."},
    {"role": "assistant", "content": "Quantum computing uses qubits." * 30},
    {"role": "user", "content": "What about entanglement?"},
    {"role": "assistant", "content": "Entanglement is a key principle." * 25},
]


SHORT_HISTORY = [
    {"role": "system", "content": "You are a helper."},
    {"role": "user", "content": "hi"},
]


class TestLevel4Summarize:
    def test_summarize_calls_llm_and_replaces_messages(self, mocker):
        mock_response = mocker.Mock()
        mock_response.choices = [
            mocker.Mock(message=mocker.Mock(content="User asked about quantum computing."))
        ]
        mock_client = mocker.Mock()
        mock_client.chat.completions.create.return_value = mock_response

        history = list(LONG_HISTORY)
        mm = MemoryManager(history)
        mm.level4_summarize(llm_client=mock_client, model="test-model")

        assert len(mm.active_history) == 2
        assert mm.active_history[0]["role"] == "system"
        assert "[摘要]" in mm.active_history[1]["content"]

    def test_summarize_skips_when_too_short(self, mocker):
        mock_client = mocker.Mock()
        history = list(SHORT_HISTORY)
        mm = MemoryManager(history)
        mm.level4_summarize(llm_client=mock_client, model="test-model")

        assert len(mm.active_history) == 2
        assert mm.active_history[0]["content"] == "You are a helper."
        assert mm.active_history[1]["content"] == "hi"
        mock_client.chat.completions.create.assert_not_called()

    def test_summarize_keeps_system_and_placeholder(self, mocker):
        mock_response = mocker.Mock()
        mock_response.choices = [
            mocker.Mock(message=mocker.Mock(content="Summary text"))
        ]
        mock_client = mocker.Mock()
        mock_client.chat.completions.create.return_value = mock_response

        history = [
            {"role": "system", "content": "You are a helper."},
            {"role": "system", "content": "[EARLY_CONTEXT_FOLDED: see early_folded.json]"},
            {"role": "user", "content": "Tell me about AI."},
            {"role": "assistant", "content": "AI is a broad field..." * 30},
        ]
        mm = MemoryManager(history)
        mm.level4_summarize(llm_client=mock_client, model="test-model")

        assert mm.active_history[0]["role"] == "system"
        assert "[EARLY_CONTEXT_FOLDED" in mm.active_history[1]["content"]
        assert "[摘要]" in mm.active_history[2]["content"]
        assert len(mm.active_history) == 3

    def test_summarize_prompt_includes_required_sections(self, mocker):
        mock_response = mocker.Mock()
        mock_response.choices = [
            mocker.Mock(message=mocker.Mock(content="mock summary"))
        ]
        mock_client = mocker.Mock()
        mock_client.chat.completions.create.return_value = mock_response

        history = list(LONG_HISTORY)
        mm = MemoryManager(history)
        mm.level4_summarize(llm_client=mock_client, model="test-model")

        call_args = mock_client.chat.completions.create.call_args
        messages_sent = call_args[1]["messages"]
        prompt = messages_sent[0]["content"]
        assert "任务目标" in prompt
        assert "关键步骤" in prompt
        assert "遗留问题" in prompt

    def test_summarize_stores_token_usage(self, mocker):
        mock_response = mocker.Mock()
        mock_response.choices = [
            mocker.Mock(message=mocker.Mock(content="mock summary"))
        ]
        mock_usage = mocker.Mock()
        mock_usage.prompt_tokens = 100
        mock_usage.completion_tokens = 50
        mock_usage.total_tokens = 150
        mock_response.usage = mock_usage
        mock_client = mocker.Mock()
        mock_client.chat.completions.create.return_value = mock_response

        history = list(LONG_HISTORY)
        mm = MemoryManager(history)
        mm.level4_summarize(llm_client=mock_client, model="test-model")
        assert mm._last_summary_usage is not None
        assert mm._last_summary_usage.prompt_tokens == 100
        assert mm._last_summary_usage.completion_tokens == 50
        assert mm._last_summary_usage.total_tokens == 150
