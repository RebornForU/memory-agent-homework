import sys
import os
import json
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.memory_manager import MemoryManager


SAMPLE_HISTORY = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi! How can I help?"},
    {"role": "user", "content": "What is ML?"},
    {"role": "assistant", "content": "Machine Learning is..."},
    {"role": "user", "content": "Thanks!"},
]


class TestLevel3Fold:
    def test_fold_early_preserves_system_at_index_0(self):
        history = list(SAMPLE_HISTORY)
        mm = MemoryManager(history)
        mm.fold_early(n=2)
        assert mm.active_history[0] == {"role": "system", "content": "You are a helpful assistant."}

    def test_fold_early_places_placeholder_at_index_1(self):
        history = list(SAMPLE_HISTORY)
        mm = MemoryManager(history)
        mm.fold_early(n=2)
        assert mm.active_history[1]["role"] == "system"
        assert "[EARLY_CONTEXT_FOLDED:" in mm.active_history[1]["content"]

    def test_fold_early_removes_correct_number_of_messages(self):
        history = list(SAMPLE_HISTORY)
        mm = MemoryManager(history)
        mm.fold_early(n=2)
        assert len(mm.active_history) == 1 + 1 + (len(history) - 1 - 2)

    def test_fold_early_writes_to_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            history = list(SAMPLE_HISTORY)
            mm = MemoryManager(history)
            mm.fold_early(n=2, output_dir=tmpdir)
            folded_path = os.path.join(tmpdir, "early_folded.json")
            assert os.path.isfile(folded_path)
            with open(folded_path) as f:
                folded = json.load(f)
            assert len(folded) == 2
            assert folded[0]["content"] == "Hello"
            assert folded[1]["content"] == "Hi! How can I help?"

    def test_unfold_restores_original(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            history = list(SAMPLE_HISTORY)
            mm = MemoryManager(history)
            mm.fold_early(n=2, output_dir=tmpdir)
            mm.unfold(folded_path=os.path.join(tmpdir, "early_folded.json"))
            assert mm.active_history == history

    def test_fold_early_n_equals_all_but_system(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            history = list(SAMPLE_HISTORY)
            mm = MemoryManager(history)
            mm.fold_early(n=100, output_dir=tmpdir)
            assert len(mm.active_history) == 2
            assert mm.active_history[0]["role"] == "system"

    def test_fold_early_n_equals_zero(self):
        history = list(SAMPLE_HISTORY)
        mm = MemoryManager(history)
        mm.fold_early(n=0)
        assert mm.active_history == history

    def test_multiple_folds_accumulate(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            history = list(SAMPLE_HISTORY)
            mm = MemoryManager(history)
            mm.fold_early(n=2, output_dir=tmpdir)
            path1 = os.path.join(tmpdir, "early_folded.json")
            mm.fold_early(n=2, output_dir=tmpdir)
            path2 = os.path.join(tmpdir, "early_folded.1.json")
            assert len(mm.active_history) == 1 + 2 + (len(history) - 1 - 2 - 2)
            mm.unfold(path2)
            mm.unfold(path1)
            assert mm.active_history == history

    def test_unfold_only_affects_placeholder_messages(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            history = list(SAMPLE_HISTORY)
            mm = MemoryManager(history)
            mm.fold_early(n=2, output_dir=tmpdir)
            placeholder = mm.active_history[1]
            mm.unfold(folded_path=os.path.join(tmpdir, "early_folded.json"))
            assert mm.active_history == history
