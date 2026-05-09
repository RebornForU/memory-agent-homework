import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.memory_manager import MemoryManager


def tool_msg(tool_call_id: str, content: str, name: str = "read_file"):
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": content,
        "name": name,
    }


class TestLevel2Dedup:
    def test_dedup_identical_tool_messages(self):
        history = [
            tool_msg("call_1", "file content here"),
            tool_msg("call_2", "file content here"),
        ]
        mm = MemoryManager(history)
        mm.level2_dedup()
        assert mm.active_history[0]["content"] == "file content here"
        assert mm.active_history[1]["content"] == "[内容同前，略]"

    def test_preserves_first_occurrence(self):
        history = [
            tool_msg("call_1", "unique result A"),
            tool_msg("call_2", "unique result B"),
            tool_msg("call_3", "unique result A"),
        ]
        mm = MemoryManager(history)
        mm.level2_dedup()
        assert mm.active_history[0]["content"] == "unique result A"
        assert mm.active_history[1]["content"] == "unique result B"
        assert mm.active_history[2]["content"] == "[内容同前，略]"

    def test_keeps_structure_fields_intact(self):
        history = [
            tool_msg("call_1", "same data", name="read_file"),
            tool_msg("call_2", "same data", name="search_file"),
        ]
        mm = MemoryManager(history)
        mm.level2_dedup()
        assert mm.active_history[0]["role"] == "tool"
        assert mm.active_history[0]["tool_call_id"] == "call_1"
        assert mm.active_history[0]["name"] == "read_file"
        assert mm.active_history[1]["role"] == "tool"
        assert mm.active_history[1]["tool_call_id"] == "call_2"
        assert mm.active_history[1]["name"] == "search_file"

    def test_no_change_when_all_unique(self):
        history = [
            tool_msg("call_1", "result A"),
            tool_msg("call_2", "result B"),
        ]
        mm = MemoryManager(history)
        mm.level2_dedup()
        assert mm.active_history[0]["content"] == "result A"
        assert mm.active_history[1]["content"] == "result B"

    def test_ignores_non_tool_messages(self):
        history = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            tool_msg("call_1", "data"),
            tool_msg("call_2", "data"),
        ]
        mm = MemoryManager(history)
        mm.level2_dedup()
        assert mm.active_history[0]["content"] == "hello"
        assert mm.active_history[1]["content"] == "hi"
        assert mm.active_history[2]["content"] == "data"
        assert mm.active_history[3]["content"] == "[内容同前，略]"

    def test_empty_history(self):
        mm = MemoryManager([])
        mm.level2_dedup()
        assert mm.active_history == []

    def test_single_tool_message(self):
        history = [tool_msg("call_1", "only one")]
        mm = MemoryManager(history)
        mm.level2_dedup()
        assert mm.active_history[0]["content"] == "only one"
