import pytest
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestProjectStructure:
    def test_src_agent_module_exists(self):
        from src.agent import Agent
        assert Agent is not None

    def test_src_memory_manager_module_exists(self):
        from src.memory_manager import MemoryManager
        assert MemoryManager is not None

    def test_requirements_txt_exists(self):
        assert os.path.isfile("requirements.txt")

    def test_dot_env_example_exists(self):
        assert os.path.isfile(".env.example")


class TestAgentInit:
    def test_agent_init_with_explicit_args(self):
        from src.agent import Agent
        agent = Agent(model="test-model", api_key="test-key", base_url="https://test.com")
        assert agent.model == "test-model"
        assert agent.api_key == "test-key"
        assert agent.base_url == "https://test.com"

    def test_agent_method_signatures_exist(self):
        from src.agent import Agent
        agent = Agent(model="m", api_key="k", base_url="u")
        assert hasattr(agent, "chat")
        assert hasattr(agent, "get_history")
        assert hasattr(agent, "set_history")
        assert hasattr(agent, "ask_meta")

    def test_agent_init_reads_env_vars(self, monkeypatch):
        monkeypatch.setenv("LLM_MODEL", "env-model")
        monkeypatch.setenv("LLM_API_KEY", "env-key")
        monkeypatch.setenv("LLM_BASE_URL", "https://env.com")
        from src.agent import Agent
        agent = Agent()
        assert agent.model == "env-model"
        assert agent.api_key == "env-key"
        assert agent.base_url == "https://env.com"

    def test_agent_explicit_overrides_env(self, monkeypatch):
        monkeypatch.setenv("LLM_MODEL", "env-model")
        monkeypatch.setenv("LLM_API_KEY", "env-key")
        monkeypatch.setenv("LLM_BASE_URL", "https://env.com")
        from src.agent import Agent
        agent = Agent(model="explicit-model")
        assert agent.model == "explicit-model"
        assert agent.api_key == "env-key"


class TestMemoryManagerInit:
    def test_memory_manager_init_with_history(self):
        from src.memory_manager import MemoryManager
        history = [{"role": "system", "content": "You are a helper."}]
        mm = MemoryManager(history)
        assert mm.active_history == history

    def test_memory_manager_deep_copy(self):
        from src.memory_manager import MemoryManager
        original = [{"role": "user", "content": "hello"}]
        mm = MemoryManager(original)
        original[0]["content"] = "modified"
        assert mm.active_history[0]["content"] == "hello"

    def test_memory_manager_method_signatures_exist(self):
        from src.memory_manager import MemoryManager
        mm = MemoryManager([])
        assert hasattr(mm, "level1_truncate")
        assert hasattr(mm, "level2_dedup")
        assert hasattr(mm, "fold_early")
        assert hasattr(mm, "unfold")
        assert hasattr(mm, "level4_summarize")
        assert hasattr(mm, "compress_all")

    def test_memory_manager_methods_return_self(self):
        from src.memory_manager import MemoryManager
        mm = MemoryManager([{"role": "user", "content": "hi"}])
        result = mm.level1_truncate()
        assert result is mm
        result = mm.level2_dedup()
        assert result is mm
        result = mm.fold_early()
        assert result is mm
        result = mm.level4_summarize(None)
        assert result is mm

    def test_memory_manager_methods_chain(self):
        from src.memory_manager import MemoryManager
        mm = MemoryManager([{"role": "user", "content": "hi"}])
        result = mm.level1_truncate().level2_dedup()
        assert result is mm
        assert result.active_history == [{"role": "user", "content": "hi"}]
