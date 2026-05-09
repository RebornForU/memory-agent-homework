import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.memory_manager import MemoryManager


def make_msg(role: str, content: str):
    return {"role": role, "content": content}


class TestLevel1Truncate:
    def test_truncates_long_assistant_message(self):
        long_text = "决策部署。" * 200
        history = [make_msg("assistant", long_text)]
        mm = MemoryManager(history)
        mm.level1_truncate(max_chars=500)
        result = mm.active_history[0]["content"]
        assert len(result) <= 500 + len("[TRUNCATED]")
        assert result.endswith("[TRUNCATED]")

    def test_safe_boundary_on_period(self):
        text_before = "A" * 490
        text_after = "B" * 100
        long_text = text_before + "。" + text_after
        history = [make_msg("assistant", long_text)]
        mm = MemoryManager(history)
        mm.level1_truncate(max_chars=500)
        result = mm.active_history[0]["content"]
        assert result.endswith("。[TRUNCATED]")
        assert "B" not in result

    def test_falls_back_to_space_when_no_punctuation(self):
        text_before = "A" * 490
        text_after = "B" * 100
        long_text = text_before + " " + text_after
        history = [make_msg("assistant", long_text)]
        mm = MemoryManager(history)
        mm.level1_truncate(max_chars=500)
        result = mm.active_history[0]["content"]
        assert "B" not in result

    def test_hard_cut_when_no_boundary(self):
        long_text = "A" * 1000
        history = [make_msg("assistant", long_text)]
        mm = MemoryManager(history)
        mm.level1_truncate(max_chars=500)
        result = mm.active_history[0]["content"]
        assert len(result) == 500 + len("[TRUNCATED]")

    def test_does_not_truncate_under_max_chars(self):
        text = "Hello, how can I help you?"
        history = [make_msg("assistant", text)]
        mm = MemoryManager(history)
        mm.level1_truncate(max_chars=500)
        assert mm.active_history[0]["content"] == text

    def test_does_not_truncate_at_exact_max_chars(self):
        text = "A" * 500
        history = [make_msg("assistant", text)]
        mm = MemoryManager(history)
        mm.level1_truncate(max_chars=500)
        assert mm.active_history[0]["content"] == text

    def test_preserves_non_assistant_messages(self):
        history = [
            make_msg("system", "You are a helper."),
            make_msg("user", "Hello"),
            make_msg("assistant", "A" * 1000),
            make_msg("tool", "result data"),
        ]
        mm = MemoryManager(history)
        mm.level1_truncate(max_chars=500)
        assert mm.active_history[0]["content"] == "You are a helper."
        assert mm.active_history[1]["content"] == "Hello"
        assert mm.active_history[2]["content"].endswith("[TRUNCATED]")
        assert mm.active_history[3]["content"] == "result data"

    def test_json_validity_after_truncation(self):
        long_text = "A" * 1000
        history = [
            make_msg("system", "helper"),
            make_msg("assistant", long_text),
        ]
        mm = MemoryManager(history)
        mm.level1_truncate(max_chars=500)
        dumped = json.dumps(mm.active_history, ensure_ascii=False)
        loaded = json.loads(dumped)
        assert loaded == mm.active_history

    def test_handles_none_content_assistant(self):
        history = [
            {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1"}]},
        ]
        mm = MemoryManager(history)
        mm.level1_truncate(max_chars=500)
        assert mm.active_history[0]["content"] is None
        assert mm.active_history[0]["tool_calls"] == [{"id": "call_1"}]

    def test_chain_with_scaffold_tests(self):
        history = [make_msg("assistant", "A" * 1000)]
        mm = MemoryManager(history)
        result = mm.level1_truncate(max_chars=500).level2_dedup()
        assert result.active_history[0]["content"].endswith("[TRUNCATED]")
