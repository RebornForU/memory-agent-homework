# PRD: 多级记忆压缩机制对话 Agent

## Problem Statement

大语言模型 Agent 在实际部署中面临一个核心约束：**上下文窗口有限**。每次与 LLM 的交互都会将对话历史（system prompt、用户消息、助手回复、工具调用及其结果）累加进 token 计数。当历史过长时，要么超出模型的最大上下文窗口导致调用失败，要么即使未超限也会因 attention 分散而严重降低回复质量。

现有解决方案（如滑动窗口、完全丢弃早期消息、或一次性对所有历史做摘要）各自存在缺陷：滑动窗口会突然丢失关键上下文；完全丢弃不可逆，无法追溯；一次性全量摘要丢失细粒度信息且无法增量更新。

因此需要一个**多级、渐进式的记忆压缩流水线**，从最廉价（截断长回复）到最昂贵（调用 LLM 做语义摘要）逐级施加，在保留任务关键信息的前提下将历史压缩到可控规模。

## Solution

构建一个 `MemoryManager` 类，对对话历史（OpenAI Chat Completion 格式的消息列表）依次施加**四级压缩**：

1. **Level 1 — 长输出截断**：对超长 assistant 消息做字符级截断，保证 JSON 合法性
2. **Level 2 — 重复工具输出去重**：对完全相同的 tool 返回做替换
3. **Level 3 — 早期对话折叠**：将最靠前的 n 条消息移出活跃历史，保存到外部文件，支持可逆恢复
4. **Level 4 — 子 Agent 摘要**：调用 LLM 对剩余历史做语义摘要，用摘要消息替换原始内容

压缩后的历史仍可被 `Agent` 类用于正常的 LLM 问答，确保 Agent 能理解当前讨论进展。

## User Stories

1. 作为开发者，我希望对超长的 assistant 回复进行截断时不破坏 JSON 结构的合法性，以便下游 json.loads 不会报错。
2. 作为开发者，我希望截断时能智能地落在最后一个完整句号等安全边界上，而不是生硬截断在字符串中间。
3. 作为开发者，我希望截断后的消息末尾有明显标记（如 `[TRUNCATED]`），以便知道该消息已被截断。
4. 作为开发者，我希望相同 tool_call_id 多次返回完全一致的内容时只保留第一次，后续替换为占位文本，以减少冗余 token。
5. 作为开发者，我希望去重后的工具消息仍然保留原有的消息结构和 tool_call_id，以便下游代码无需改动即可正常解析。
6. 作为开发者，我希望能够将早期的 n 条对话历史折叠为一个占位符消息，并将其原始内容保存到外部 JSON 文件中。
7. 作为开发者，我希望折叠操作是可逆的：调用 `unfold()` 方法后能精确恢复被折叠的原始消息，包括 system 设定和所有角色消息。
8. 作为开发者，我希望占位符消息中包含明确的文件引用路径，以便在需要时可以手动找到被折叠的内容。
9. 作为开发者，我希望经过前三级压缩后，能调用同一个 LLM 对剩余历史做一次语义摘要，进一步压缩 token 占用。
10. 作为开发者，我希望摘要消息能保留任务进度、遗留问题、关键数据等核心信息，而不是简单的概括。
11. 作为开发者，我希望摘要后的对话历史仍能用于 LLM 问答，Agent 能正确回答"我们现在在讨论什么"这类元问题。
12. 作为开发者，我希望能在 Jupyter Notebook 中逐级观察每级压缩前后的消息变化，方便调试和分析。
13. 作为开发者，我希望有量化指标（如压缩率 = 压缩后 token 数 / 压缩前 token 数）来评估每级压缩的实际效果。

## Implementation Decisions

### 模块划分

系统由两个核心类构成：

#### `Agent` 类
- 职责：封装 LLM API 调用逻辑（支持阿里百炼或 OpenRouter），维护消息历史，调用 `MemoryManager` 执行压缩。
- 接口：
  - `__init__(self, model: str, api_key: str, base_url: str)`
  - `chat(self, user_message: str) -> str`：接收用户输入，追加到历史，调用 LLM，返回回复
  - `get_history(self) -> list[dict]`：返回当前完整历史
  - `set_history(self, history: list[dict])`：设置历史（用于压缩后回写）
  - `ask_meta(self, question: str) -> str`：用当前历史回答元问题

#### `MemoryManager` 类
- 职责：纯函数式/无状态地执行四级压缩流水线
- 接口：
  - `__init__(self, history: list[dict])`
  - `level1_truncate(max_chars: int = 500) -> list[dict]`：长输出截断
  - `level2_dedup() -> list[dict]`：重复工具输出去重
  - `fold_early(n: int = 10, output_dir: str = ".") -> list[dict]`：折叠前 n 条
  - `unfold(folded_path: str) -> list[dict]`：从文件恢复
  - `level4_summarize(llm_client, model: str) -> list[dict]`：子 Agent 摘要
  - `compress_all() -> list[dict]`：依次执行全部四级压缩

### 各层级技术决策

#### Level 1: 长输出截断（强调 JSON 有效性）
- **检测范围**：仅作用于 `role == "assistant"` 且 `len(content) > 500` 的消息。
- **截断策略**：保留前 n 个字符（具体 n 可由调用者指定，默认 n=500），从位置 n 开始向前搜索最后一个句号（`。`或 `.`）、感叹号、问号、换行符等安全边界。
- **如果找不到任何安全边界**，则退回到 n 位置向前搜索最后一个空格，仍找不到则按 n 硬切（这种情况极罕见，仅发生在超长无标点文本中）。
- **标记**：在截断后的末尾追加 `...[TRUNCATED]`。
- **JSON 有效性验证**：截断后使用 `json.dumps(msg)` + `json.loads(json.dumps(msg))` 验证整条消息列表可被合法解析。这是出厂检验逻辑。
- **边界情况**：对于 tool 消息中的长 content 也可选做截断（本次作业不要求，但作为扩展点记录）。

#### Level 2: 重复工具输出去重
- **判定条件**：严格相等（`==`），比较两个字符串的每个字符。
- **匹配范围**：仅对 `role == "tool"` 的消息进行。
- **替换方式**：将 `content` 替换为 `"[内容同前，略]"`。保留 `tool_call_id`、`role`、`name` 等字段原样不变，确保消息结构完整。

#### Level 3: 早期对话折叠（强调可逆性）
- **折叠逻辑**：
  - `fold_early(n)`：截取 `history[:n]`，写入文件 `early_folded.json`，然后从活跃历史中移除这 n 条，在原来位置（索引 0）插入占位符消息。
  - 占位符格式：`{"role": "system", "content": "[EARLY_CONTEXT_FOLDED: see early_folded.json]"}`。
  - 第一条 system 消息建议保留在活跃历史中（即从索引 1 开始折叠），或者将其一起折叠（取决于 n 的取值）。**推荐 n >= 10 时覆盖第一条 system 消息，此时占位符 role 用 "system"；若折叠的消息中不包含 system，则占位符 role 可以用 "user"**。
- **恢复逻辑**：
  - `unfold()`：扫描历史中所有 content 匹配 `[EARLY_CONTEXT_FOLDED*` 的消息，读取 `early_folded.json`，将占位符替换为原始消息列表（展开插入）。
  - 支持多次折叠的累积恢复。
- **验证方法**：`assert unfold(fold_early(n)) == original_history`。

#### Level 4: 子 Agent 摘要
- **提取范围**：经过前三级的消息列表中，排除 role == "system" 的消息和占位符消息。
- **摘要提示词**（可完善）：
  ```
  你是一个对话历史摘要助手。请用 200 token 以内总结以下对话的要点。
  要求：
  1. 保留任务目标与当前进度
  2. 列出已经完成的关键步骤
  3. 列出尚未解决的遗留问题或下一步计划
  4. 保留所有重要的数据、数值和配置信息
  5. 不添加原始对话中没有的信息

  对话内容：
  {messages_text}
  ```
- **插入位置**：生成一条 `{"role": "system", "content": "[摘要] {summary}"}` 消息，插入到活跃历史中（建议放在第一条 system 或占位符之后），然后从历史中移除被摘要的那些消息。
- **API 成本控制**：摘要使用通过 `Agent` 的同一个 LLM 客户端调用，但可以使用更小的模型（如 qwen-turbo）以降低成本。可选的成本优化：如果被摘要文本总 token 数低于阈值（如 500 tokens），则跳过摘要。

### 对话数据格式

输入/输出使用 OpenAI Chat Completion 格式：
```json
[
  {"role": "system", "content": "You are a helpful assistant."},
  {"role": "user", "content": "Hello"},
  {"role": "assistant", "content": "Hi! How can I help you?", "tool_calls": [...]},
  {"role": "tool", "content": "Result", "tool_call_id": "call_xxx", "name": "read_file"}
]
```

### 压缩率评估方法

使用 `len(json.dumps(messages, ensure_ascii=False))` 粗略估算字符数（可作为 token 数的线性代理指标）。

压缩率公式：
```
compression_ratio = compressed_size / original_size
```

每级压缩后记录该级压缩率，以及累积压缩率。

## Testing Decisions

### 测试原则
- 测试应聚焦于外部行为（输入消息列表 → 输出压缩后的消息列表），而非内部实现细节。
- 每个压缩层级独立测试，确保可单独验证正确性。
- 第三级折叠的可逆性需要专门的单元测试验证。

### 测试模块

| 测试目标 | 方法 |
|---------|------|
| Level 1 截断 | 构造超长 assistant 回复（含标点、不含标点、恰好 500 字符），验证截断长度 ≤ max_chars + len("[TRUNCATED]")，验证 JSON 可解析 |
| Level 1 安全边界 | 构造标点在 n 前后的内容，验证截断落在最后一个句号而非 n 位置 |
| Level 2 去重 | 构造多条完全相同和部分不同的 tool 消息，验证只有第一条被保留，其余被替换 |
| Level 2 结构完整性 | 验证去重后的每条 tool 消息仍包含 role、content、tool_call_id、name 字段 |
| Level 3 折叠 | 折叠前 n 条后验证活跃历史只有 1 条占位符 + 剩余消息，且外部文件内容与折叠前 n 条完全一致 |
| Level 3 可逆性 | `assert unfold(fold_early(n)) == original_history` |
| Level 4 摘要 | 调用 mock LLM 返回固定摘要，验证摘要消息被正确插入且原始消息被移除 |
| 端到端流水线 | 使用完整数据依次执行四级压缩，验证最终消息列表的合法性 |

### 测试数据
- 使用 `dialogue.json` 作为主测试数据
- 为 Level 1 单独构造超长 assistant 消息测试用例
- 为 Level 2 构造带重复 tool 调用的测试用例

## Out of Scope

- 增量/在线压缩（每次新消息后自动压缩）：本次只做批量离线压缩，不在 chat 循环中自动触发。
- 多轮对话中逐轮自适应压缩策略选择（如根据当前 token 使用率决定是否需要压缩下一级）。
- 不同于 OpenAI 格式的其他消息格式（如 Anthropic Messages API）。
- 持久化存储完整历史到数据库：只使用 JSON 文件存储折叠内容。
- 压缩后的历史直接输入给 LLM 的效果评测（如 benchmark 指标）：仅做人工定性评估。
- 非中文/非英文的多语言截断安全边界（如日语、阿拉伯语）。

## Further Notes

- 实验报告应在 Jupyter Notebook 中完成，包含完整的类定义、每级压缩的结果展示、压缩率分析和最终问答验证。
- 建议使用 `json.dumps(..., indent=2, ensure_ascii=False)` 在 Notebook 中格式化打印消息列表以便观察。
- API 调用需提供 key 的配置方式（环境变量或配置文件），不应硬编码在代码中。
- 建议在 Notebook 中使用 `textwrap` 或类似工具控制长文本的打印宽度，提高可读性。
- 实验报告中应包含对 API 调用次数和 token 使用量的统计。
