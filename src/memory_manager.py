import copy
import json
import os


class MemoryManager:
    def __init__(self, history: list[dict]):
        self.active_history = copy.deepcopy(history)
        self._fold_counter = 0
        self._last_summary_usage = None

    @staticmethod
    def _find_safe_boundary(text: str, max_chars: int) -> int:
        segment = text[:max_chars]
        for delim in ("。", ".", "!", "?", "\n"):
            pos = segment.rfind(delim)
            if pos != -1:
                return pos + 1
        pos = segment.rfind(" ")
        if pos != -1:
            return pos
        return max_chars

    def level1_truncate(self, max_chars: int = 500):
        for msg in self.active_history:
            if msg.get("role") == "assistant" and isinstance(msg.get("content"), str):
                content = msg["content"]
                if len(content) > max_chars:
                    boundary = self._find_safe_boundary(content, max_chars)
                    msg["content"] = content[:boundary] + "[TRUNCATED]"
        return self

    def level2_dedup(self):
        seen = set()
        for msg in self.active_history:
            if msg.get("role") == "tool" and isinstance(msg.get("content"), str):
                content = msg["content"]
                if content in seen:
                    msg["content"] = "[内容同前，略]"
                else:
                    seen.add(content)
        return self

    def fold_early(self, n: int = 10, output_dir: str = "."):
        if n <= 0 or len(self.active_history) <= 1:
            return self
        n = min(n, len(self.active_history) - 1)
        folded = self.active_history[1 : 1 + n]
        self._fold_counter += 1
        if self._fold_counter == 1:
            filename = "early_folded.json"
        else:
            filename = f"early_folded.{self._fold_counter - 1}.json"
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(folded, f, ensure_ascii=False, indent=2)
        placeholder = {
            "role": "system",
            "content": f"[EARLY_CONTEXT_FOLDED: see {filename}]",
        }
        self.active_history = (
            [self.active_history[0]]
            + [placeholder]
            + self.active_history[1 + n :]
        )
        return self

    def unfold(self, folded_path: str):
        with open(folded_path, encoding="utf-8") as f:
            folded = json.load(f)
        filename = os.path.basename(folded_path)
        new_history = []
        for msg in self.active_history:
            if (
                isinstance(msg.get("content"), str)
                and f"[EARLY_CONTEXT_FOLDED: see {filename}]" in msg["content"]
            ):
                new_history.extend(folded)
            else:
                new_history.append(msg)
        self.active_history = new_history
        return self

    def level4_summarize(self, llm_client, model: str | None = None):
        keep_indices = set()
        summarize_indices = []
        for i, msg in enumerate(self.active_history):
            if msg.get("role") == "system":
                keep_indices.add(i)
            elif isinstance(msg.get("content"), str) and "[EARLY_CONTEXT_FOLDED:" in msg["content"]:
                keep_indices.add(i)
            else:
                summarize_indices.append(i)

        if not summarize_indices:
            return self

        to_summarize = [self.active_history[i] for i in summarize_indices]
        text_to_summarize = "\n".join(
            f"{m['role']}: {m['content']}" for m in to_summarize
        )

        if len(text_to_summarize) <= 500:
            return self

        prompt = (
            "你是一个对话历史摘要助手。请用 200 token 以内总结以下对话的要点。\n"
            "要求：\n"
            "1. 保留任务目标与当前进度\n"
            "2. 列出已经完成的关键步骤\n"
            "3. 列出尚未解决的遗留问题或下一步计划\n"
            "4. 保留所有重要的数据、数值和配置信息\n"
            "5. 不添加原始对话中没有的信息\n\n"
            f"对话内容：\n{text_to_summarize}"
        )

        response = llm_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model,
        )
        self._last_summary_usage = response.usage
        summary = response.choices[0].message.content

        summary_msg = {"role": "system", "content": f"[摘要] {summary}"}

        new_history = [
            self.active_history[i] for i in sorted(keep_indices)
        ]
        insert_pos = len(new_history)
        new_history.insert(insert_pos, summary_msg)
        self.active_history = new_history
        return self

    def compress_all(self, max_chars: int = 500, fold_n: int = 10,
                      output_dir: str = ".", llm_client=None, model: str | None = None):
        metrics = {"original_size": 0, "levels": []}

        def record(level_name):
            size = len(json.dumps(self.active_history, ensure_ascii=False))
            if not metrics["original_size"]:
                metrics["original_size"] = size
            level_ratio = size / metrics["original_size"]
            metrics["levels"].append({
                "level": level_name,
                "size": size,
                "ratio": round(level_ratio, 4),
            })

        record("original")
        self.level1_truncate(max_chars=max_chars)
        record("level1_truncate")
        self.level2_dedup()
        record("level2_dedup")
        self.fold_early(n=fold_n, output_dir=output_dir)
        record("level3_fold")
        if llm_client is not None:
            self.level4_summarize(llm_client=llm_client, model=model)
            record("level4_summarize")
        return metrics
