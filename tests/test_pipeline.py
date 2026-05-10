import sys
import os
import json
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.memory_manager import MemoryManager
from src.agent import Agent


HISTORY_FIXTURE = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is ML?"},
    {"role": "assistant", "content": "Machine Learning is..." * 50},
    {"role": "user", "content": "Tell me more."},
    {"role": "assistant", "content": "Sure! ML has many applications." * 50},
]


class TestCompressAll:
    def test_compress_all_runs_without_llm(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            history = list(HISTORY_FIXTURE)
            mm = MemoryManager(history)
            metrics = mm.compress_all(output_dir=tmpdir)
            assert "original_size" in metrics
            assert len(metrics["levels"]) >= 4
            assert metrics["levels"][0]["level"] == "original"
            assert metrics["levels"][1]["level"] == "level1_truncate"
            assert metrics["levels"][2]["level"] == "level2_dedup"
            assert metrics["levels"][3]["level"] == "level3_fold"

    def test_compress_all_with_llm(self, mocker):
        mock_response = mocker.Mock()
        mock_response.choices = [
            mocker.Mock(message=mocker.Mock(content="Summary about ML."))
        ]
        mock_client = mocker.Mock()
        mock_client.chat.completions.create.return_value = mock_response

        with tempfile.TemporaryDirectory() as tmpdir:
            history = list(HISTORY_FIXTURE)
            mm = MemoryManager(history)
            metrics = mm.compress_all(
                output_dir=tmpdir, llm_client=mock_client, model="test"
            )
            assert len(metrics["levels"]) == 5

    def test_compress_all_with_dialogue_json(self):
        dialogue_path = os.path.join(
            os.path.dirname(__file__), "..", "dialogue.json"
        )
        with open(dialogue_path, encoding="utf-8") as f:
            history = json.load(f)
        with tempfile.TemporaryDirectory() as tmpdir:
            mm = MemoryManager(history)
            metrics = mm.compress_all(output_dir=tmpdir)
            assert metrics["levels"][-1]["ratio"] < 1.0


class TestAgent:
    def test_agent_chat_calls_llm_and_updates_history(self, mocker):
        mock_response = mocker.Mock()
        mock_response.choices = [
            mocker.Mock(message=mocker.Mock(content="Hello! How can I help?"))
        ]
        mock_client = mocker.Mock()
        mock_client.chat.completions.create.return_value = mock_response

        agent = Agent(model="test", api_key="fake", base_url="https://fake.com")
        agent.client = mock_client
        reply = agent.chat("Hi there")
        assert reply == "Hello! How can I help?"
        assert len(agent.get_history()) == 2
        assert agent.get_history()[0]["role"] == "user"
        assert agent.get_history()[1]["role"] == "assistant"

    def test_agent_set_and_get_history(self):
        agent = Agent(model="test", api_key="fake", base_url="https://fake.com")
        history = [{"role": "user", "content": "hello"}]
        agent.set_history(history)
        assert agent.get_history() == history

    def test_agent_ask_meta_calls_llm(self, mocker):
        mock_response = mocker.Mock()
        mock_response.choices = [
            mocker.Mock(message=mocker.Mock(content="We discussed ML."))
        ]
        mock_client = mocker.Mock()
        mock_client.chat.completions.create.return_value = mock_response

        agent = Agent(model="test", api_key="fake", base_url="https://fake.com")
        agent.client = mock_client
        agent.set_history([{"role": "user", "content": "Tell me about ML."}])
        reply = agent.ask_meta("What did we discuss?")
        assert reply == "We discussed ML."

    def test_agent_tracks_token_usage_on_chat(self, mocker):
        mock_response = mocker.Mock()
        mock_response.choices = [
            mocker.Mock(message=mocker.Mock(content="Hi"))
        ]
        mock_usage = mocker.Mock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 5
        mock_usage.total_tokens = 15
        mock_response.usage = mock_usage
        mock_client = mocker.Mock()
        mock_client.chat.completions.create.return_value = mock_response

        agent = Agent(model="test", api_key="fake", base_url="https://fake.com")
        agent.client = mock_client
        agent.chat("Hello")
        assert agent.total_usage["prompt_tokens"] == 10
        assert agent.total_usage["completion_tokens"] == 5
        assert agent.total_usage["total_tokens"] == 15
        assert agent.total_usage["calls"] == 1

    def test_agent_accumulates_token_usage_across_calls(self, mocker):
        mock_response = mocker.Mock()
        mock_response.choices = [
            mocker.Mock(message=mocker.Mock(content="Hi"))
        ]
        mock_usage = mocker.Mock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 5
        mock_usage.total_tokens = 15
        mock_response.usage = mock_usage
        mock_client = mocker.Mock()
        mock_client.chat.completions.create.return_value = mock_response

        agent = Agent(model="test", api_key="fake", base_url="https://fake.com")
        agent.client = mock_client
        agent.chat("Q1")
        agent.ask_meta("Q2")
        assert agent.total_usage["prompt_tokens"] == 20
        assert agent.total_usage["completion_tokens"] == 10
        assert agent.total_usage["total_tokens"] == 30
        assert agent.total_usage["calls"] == 2

    def test_agent_compress_history_reduces_history_size(self, mocker):
        mock_response = mocker.Mock()
        mock_response.choices = [
            mocker.Mock(message=mocker.Mock(content="Mock summary."))
        ]
        mock_client = mocker.Mock()
        mock_client.chat.completions.create.return_value = mock_response

        history = list(HISTORY_FIXTURE)
        agent = Agent(model="test", api_key="fake", base_url="https://fake.com")
        agent.client = mock_client
        agent.set_history(history)

        before = len(json.dumps(agent.get_history(), ensure_ascii=False))
        with tempfile.TemporaryDirectory() as tmpdir:
            result = agent.compress_history(output_dir=tmpdir)
        after = len(json.dumps(agent.get_history(), ensure_ascii=False))

        assert after < before
        assert "metrics" in result

    def test_agent_compress_history_skip_fold(self, mocker):
        mock_response = mocker.Mock()
        mock_response.choices = [
            mocker.Mock(message=mocker.Mock(content="Mock summary."))
        ]
        mock_client = mocker.Mock()
        mock_client.chat.completions.create.return_value = mock_response

        history = list(HISTORY_FIXTURE)
        agent = Agent(model="test", api_key="fake", base_url="https://fake.com")
        agent.client = mock_client
        agent.set_history(history)

        with tempfile.TemporaryDirectory() as tmpdir:
            agent.compress_history(skip_fold=True, output_dir=tmpdir)

        folded_markers = [
            m for m in agent.get_history()
            if isinstance(m.get("content"), str)
            and "EARLY_CONTEXT_FOLDED" in m["content"]
        ]
        assert len(folded_markers) == 0
