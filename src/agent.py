import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)

_no_proxy_from_dotenv = os.getenv("NO_PROXY", "")
if _no_proxy_from_dotenv:
    for _var in ("no_proxy", "NO_PROXY"):
        _cur = os.environ.get(_var, "")
        if _no_proxy_from_dotenv not in _cur:
            os.environ[_var] = f"{_cur},{_no_proxy_from_dotenv}" if _cur else _no_proxy_from_dotenv


class Agent:
    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        self.model = model or os.getenv("LLM_MODEL")
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self.base_url = base_url or os.getenv("LLM_BASE_URL")
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.history: list[dict] = []
        self.total_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "calls": 0,
        }

    def _record_usage(self, response):
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        pt = getattr(usage, "prompt_tokens", None)
        ct = getattr(usage, "completion_tokens", None)
        tt = getattr(usage, "total_tokens", None)
        if isinstance(pt, (int, float)) and isinstance(ct, (int, float)) and isinstance(tt, (int, float)):
            self.total_usage["prompt_tokens"] += pt
            self.total_usage["completion_tokens"] += ct
            self.total_usage["total_tokens"] += tt
            self.total_usage["calls"] += 1

    def chat(self, user_message: str) -> str:
        self.history.append({"role": "user", "content": user_message})
        response = self.client.chat.completions.create(
            messages=self.history,
            model=self.model,
        )
        self._record_usage(response)
        reply = response.choices[0].message.content
        self.history.append({"role": "assistant", "content": reply})
        return reply

    def get_history(self) -> list[dict]:
        return self.history

    def set_history(self, history: list[dict]) -> None:
        self.history = history

    def ask_meta(self, question: str) -> str:
        meta_messages = self.history + [
            {"role": "user", "content": f"基于我们的对话历史，请回答：{question}"}
        ]
        response = self.client.chat.completions.create(
            messages=meta_messages,
            model=self.model,
        )
        self._record_usage(response)
        return response.choices[0].message.content

    def compress_history(self, max_chars: int = 500, fold_n: int = 10,
                          output_dir: str = ".", skip_fold: bool = False):
        from src.memory_manager import MemoryManager
        mm = MemoryManager(self.history)
        metrics = mm.compress_all(
            max_chars=max_chars,
            fold_n=fold_n,
            output_dir=output_dir,
            llm_client=self.client,
            model=self.model,
            skip_fold=skip_fold,
        )
        self.set_history(mm.active_history)
        return {"metrics": metrics, "memory_manager": mm}
