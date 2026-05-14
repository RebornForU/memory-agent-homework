#!/usr/bin/env python
# coding: utf-8

# # 多级记忆压缩机制对话 Agent — 实验报告
# 
# ## 总体思路
# 
# 大语言模型 Agent 在实际部署中面临上下文窗口有限的约束。随着对话轮次累积，消息列表的 token 数持续增长，可能超出模型最大上下文窗口或导致 reply 质量因 attention 分散而下降。
# 
# 本实验设计了一个**四级渐进式记忆压缩流水线**，从最廉价到最昂贵逐级施加压缩，在保留任务关键信息的前提下将对话历史压缩至可控规模：
# 
# | 级别 | 名称 | 策略 | 代价 |
# |------|------|------|------|
# | Level 1 | 长输出截断 | 对超长 assistant 消息做字符级安全截断 | $O(n)$ 字符串扫描 |
# | Level 2 | 重复工具去重 | 完全相同的 tool 输出替换为占位符 | $O(n)$ set 查找 |
# | Level 3 | 早期对话折叠 | 将前 n 条移出活跃历史到外部文件 | 磁盘 I/O |
# | Level 4 | 子 Agent 摘要 | 调用 LLM 对剩余历史做语义摘要 | LLM API 调用 |
# 
# 压缩后的历史仍可被 `Agent` 类用于正常的 LLM 问答，确保 Agent 能理解当前讨论进展。

# In[ ]:


import sys, os, json, tempfile
sys.path.insert(0, '.')
from src.memory_manager import MemoryManager
from src.agent import Agent

def size_of(history):
    return len(json.dumps(history, ensure_ascii=False))


# ## 类定义代码
# 
# 以下是本实验的核心类实现代码。

# In[ ]:


import textwrap

try:
    with open('src/agent.py') as f:
        agent_code = f.read()
    with open('src/memory_manager.py') as f:
        mm_code = f.read()
    print("=" * 50, "Agent 类", "=" * 50)
    print(agent_code)
    print("=" * 50, "MemoryManager 类", "=" * 50)
    print(mm_code)
except FileNotFoundError:
    print("（src/agent.py 或 src/memory_manager.py 不在当前目录）")


# ## 1. 加载原始对话数据

# In[ ]:


with open('dialogue.json', encoding='utf-8') as f:
    original_history = json.load(f)

print(f"原始消息数: {len(original_history)}")
print(f"原始字符数: {size_of(original_history)}")


# ## 2. Level 1 — 长输出截断
# 
# 对 `role == "assistant"` 且 `len(content) > 500` 的消息，从第 500 字符向前搜索安全边界（。.!?\n → 空格 → 硬切），追加 `[TRUNCATED]`。

# In[ ]:


mm = MemoryManager(original_history)

long_msgs_before = {i: m['content'] for i, m in enumerate(original_history)
                    if m.get('role') == 'assistant' and isinstance(m.get('content'), str) and len(m['content']) > 500}

before = size_of(mm.active_history)
mm.level1_truncate(max_chars=500)
after = size_of(mm.active_history)

print(f"截断前: {before} chars")
print(f"截断后: {after} chars")
print(f"压缩率: {after/before:.2%}")

truncated = [m for m in mm.active_history if isinstance(m.get('content'), str) and m['content'].endswith('[TRUNCATED]')]
print(f"被截断的消息数: {len(truncated)}")

if truncated and long_msgs_before:
    idx, orig_text = list(long_msgs_before.items())[0]
    new_text = mm.active_history[idx]['content']
    print()
    print("=" * 60)
    print("截断示例")
    print("=" * 60)
    print(f"【截断前原文（截取前 600 chars 展示对比）】")
    print(textwrap.fill(orig_text[:600], width=80))
    print(f"\n... (后续还有很长，原文共 {len(orig_text)} chars)")
    print(f"\n【截断后实际内容（全量展示，共 {len(new_text)} chars）】")
    print(new_text)


# ## 3. Level 2 — 重复工具去重
# 
# 对 `role == "tool"` 的消息做严格相等（`==`）比较。相同内容的第一次出现保留原文，后续替换为 `[内容同前，略]`，保留 role、tool_call_id、name 字段不变。

# In[ ]:


# Find first duplicate content before dedup
tool_contents = [m['content'] for m in mm.active_history if m.get('role') == 'tool' and isinstance(m.get('content'), str)]
duplicate_content = None
for c in tool_contents:
    if tool_contents.count(c) > 1:
        duplicate_content = c
        break

before = size_of(mm.active_history)
mm.level2_dedup()
after = size_of(mm.active_history)

print(f"去重前: {before} chars")
print(f"去重后: {after} chars")
print(f"压缩率: {after/before:.2%}")

deduped_count = sum(1 for m in mm.active_history if m.get('content') == '[内容同前，略]')
print(f"被去重的消息数: {deduped_count}")

if duplicate_content:
    print()
    print("=" * 60)
    print("去重示例")
    print("=" * 60)
    print(f"【原始内容（首次出现，已保留）】")
    print(duplicate_content[:200] + "..." if len(duplicate_content) > 200 else duplicate_content)
    print(f"\n【第2次出现，替换为】")
    print('[内容同前，略]')


# ## 4. Level 3 — 早期对话折叠
# 
# 保留 `history[0]`（system 消息）不动，从索引 1 开始折叠 n 条到 `early_folded.json`，在索引 1 插入占位符 `[EARLY_CONTEXT_FOLDED: see ...]`。调用 `unfold()` 可精确恢复。

# In[ ]:


with tempfile.TemporaryDirectory() as tmpdir:
    mm_fold = MemoryManager(list(mm.active_history))
    before = size_of(mm_fold.active_history)

    print("折叠前活跃历史（前 3 条）:")
    for msg in mm_fold.active_history[:3]:
        c = msg.get('content', '')[:80]
        print(f"  [{msg['role']}] {c}..." if len(msg.get('content', '')) > 80 else f"  [{msg['role']}] {c}")
    print()

    mm_fold.fold_early(n=10, output_dir=tmpdir)
    after = size_of(mm_fold.active_history)

    print(f"折叠前: {before} chars")
    print(f"折叠后: {after} chars")
    print(f"压缩率: {after/before:.2%}")
    print()
    print("折叠后活跃历史（前 3 条）:")
    for msg in mm_fold.active_history[:3]:
        c = msg.get('content', '')[:80]
        print(f"  [{msg['role']}] {c}..." if len(msg.get('content', '')) > 80 else f"  [{msg['role']}] {c}")

    folded_path = os.path.join(tmpdir, 'early_folded.json')
    with open(folded_path) as f:
        folded = json.load(f)
    print()
    print("=" * 60)
    print("被折叠内容预览（json.dumps 格式化）")
    print(f"（共 {len(folded)} 条消息，保存至 early_folded.json）")
    print("=" * 60)
    for msg in folded[:2]:
        print(json.dumps(msg, indent=2, ensure_ascii=False))
    if len(folded) > 2:
        print(f"\n  ... 还有 {len(folded) - 2} 条消息（省略）")


# ### 折叠可逆性验证

# In[ ]:


with tempfile.TemporaryDirectory() as tmpdir:
    original = list(original_history)
    mm2 = MemoryManager(original)
    mm2.fold_early(n=10, output_dir=tmpdir)
    mm2.unfold(os.path.join(tmpdir, 'early_folded.json'))
    is_equal = mm2.active_history == original
    print(f"unfold(fold_early()) == original_history: {is_equal}")


# ## 5. Level 4 — 子 Agent 摘要
# 
# 调用同一个 LLM 对剩余历史做语义摘要。排除 system 和占位符消息，若被摘要文本 ≤ 500 字符则跳过（成本优化）。

# In[ ]:


llm_available = os.getenv('LLM_API_KEY') and os.getenv('LLM_BASE_URL')
if llm_available:
    agent = Agent()

    summarize_text = []
    for m in mm.active_history:
        if m.get('role') != 'system' and '[EARLY_CONTEXT_FOLDED' not in str(m.get('content', '')):
            summarize_text.append(f"[{m['role']}] {m.get('content', '')}")
    summary_input = '\n'.join(summarize_text)

    before = size_of(mm.active_history)
    mm.level4_summarize(llm_client=agent.client, model=agent.model)
    after = size_of(mm.active_history)

    print(f"摘要前: {before} chars")
    print(f"摘要后: {after} chars")
    print(f"压缩率: {after/before:.2%}")
    print()
    print("=" * 60)
    print(f"摘要前原文片段（截取前 1200 chars，总计 {len(summary_input)} chars）")
    print("=" * 60)
    print(textwrap.fill(summary_input[:1200], width=80))
    remaining = len(summary_input) - 1200
    if remaining > 0:
        print(f"\n... [此处省略剩余 {remaining} 字符] ...")
    print()
    print("=" * 60)
    print("摘要结果")
    print("=" * 60)
    for msg in mm.active_history:
        c = msg.get('content', '')
        if '[摘要]' in c:
            print(c)
else:
    print("LLM 未配置，跳过 Level 4。")


# ## 6. 端到端流水线 & 压缩率汇总

# In[ ]:


client = agent.client if llm_available else None

with tempfile.TemporaryDirectory() as tmpdir:
    mm3 = MemoryManager(original_history)
    metrics = {'original_size': 0, 'levels': []}

    def record(level_name):
        size = len(json.dumps(mm3.active_history, ensure_ascii=False))
        if not metrics['original_size']:
            metrics['original_size'] = size
        metrics['levels'].append({
            'level': level_name,
            'size': size,
        })

    record('original')
    mm3.level1_truncate(max_chars=500)
    record('level1_truncate')
    mm3.level2_dedup()
    record('level2_dedup')

    mm3.fold_early(n=10, output_dir=tmpdir)
    record('level3_fold')

    if llm_available:
        mm3.level4_summarize(llm_client=client, model=agent.model)
        record('level4_summarize')

from IPython.display import display, Markdown

lines = []
lines.append('### 端到端压缩率汇总 （L1 → L2 → L3 → L4 完整流水线）')
lines.append('')
lines.append('| 层级 | 输入字符 | 输出字符 | 估算 Tokens | 单级压缩率 | 累积压缩率 |')
lines.append('|------|---------|---------|------------|-----------|-----------|')

original_size = metrics['original_size']
prev_size = original_size
for entry in metrics['levels']:
    level = entry['level']
    size = entry['size']
    est = f'~{int(size / 3.5)}'
    single = f'{size / prev_size:.2%}' if prev_size else '-'
    cum = f'{size / original_size:.2%}'
    lines.append(f'| {level} | {prev_size} | {size} | {est} | {single} | {cum} |')
    prev_size = size

final_ratio = metrics['levels'][-1]['size'] / original_size
lines.append('')
lines.append(f'**最终压缩率: {final_ratio:.2%}**')
display(Markdown(chr(10).join(lines)))


# ## 6.5 压缩率分析
# 
# 本次实验使用 `len(json.dumps(..., ensure_ascii=False))` 作为 token 数的粗略估算代理（中英文混合文本约 3.5 chars ≈ 1 token）。
# 
# ### 各级压缩效果分析
# 
# **L1 长输出截断**: 将原始 16888 chars 压缩至 14681 chars（-13.07%）。原始对话中存在 2 条超长 assistant 回复，截断算法从第 500 字符处向前搜索安全边界并添加 `[TRUNCATED]` 标记。
# 
# **L2 重复工具去重**: 将 14681 chars 压缩至 7883 chars（-46.32%）。3 条完全相同的 tool 消息（iris.csv 数据）合并为 1 条原文 + 2 条占位符。
# 
# **L3 早期对话折叠**: 将 7883 chars 压缩至 2745 chars（-65.18%）。前 10 条早期对话（涵盖作业需求描述和初步解答）被折叠到 `early_folded.json` 文件，活跃历史中仅保留占位符和后续消息。
# 
# **L4 子 Agent 摘要**: 将 2745 chars 压缩至最终值。LLM 对 L3 折叠后剩余的对话（涉及决策树原理、可视化代码、PyTorch 对比等）做语义摘要，保留任务目标和关键进度。
# 
# ### 累积压缩效果
# 
# 四级流水线最终将 16888 chars 的对话压缩至约 300-600 chars（受 LLM 输出影响略有浮动），压缩率约 **2-3%**，同时保留了完整对话上下文。
# 

# ## 7. 元问题验证
# 
# 基于压缩后的历史，向 Agent 提问验证语义保留。

# In[ ]:


if llm_available:
    agent2 = Agent()
    agent2.set_history(mm3.active_history if 'mm3' in dir() else original_history)

    questions = [
        "我们现在在讨论什么？",
        "我们完成了哪些关键步骤？",
        "用户最后问了什么问题？",
    ]
    for q in questions:
        print(f"\nQ: {q}")
        answer = agent2.ask_meta(q)
        print(f"A: {answer}")
else:
    print("LLM 未配置，跳过元问题验证。")


# ## 8. API 调用与 Token 统计分析

# In[ ]:


if llm_available:
    # 收集所有 token 用量
    summary_usage = getattr(mm3, '_last_summary_usage', None)
    meta_usage = agent2.total_usage

    print("### Token 用量明细 ###")
    print(f"{'调用目的':<16} {'Prompt':>8} {'Completion':>12} {'Total':>8}")
    print("-" * 48)

    # Level 4 摘要（压缩阶段的那个 agent 没有记录，需要从 mm3 读取）
    if summary_usage and hasattr(summary_usage, 'prompt_tokens'):
        print(f"{'Level 4 摘要':<16} {summary_usage.prompt_tokens:>8} {summary_usage.completion_tokens:>12} {summary_usage.total_tokens:>8}")
        sum_pt = summary_usage.prompt_tokens
        sum_ct = summary_usage.completion_tokens
        sum_tt = summary_usage.total_tokens
    else:
        sum_pt = sum_ct = sum_tt = 0
        print(f"{'Level 4 摘要':<16} {'-':>8} {'-':>12} {'-':>8}")

    # 元问题（agent2 累计了 3 次 ask_meta 调用）
    print(f"{'元问题 x3':<16} {meta_usage['prompt_tokens']:>8} {meta_usage['completion_tokens']:>12} {meta_usage['total_tokens']:>8}")

    # 合计
    total_pt = sum_pt + meta_usage['prompt_tokens']
    total_ct = sum_ct + meta_usage['completion_tokens']
    total_tt = sum_tt + meta_usage['total_tokens']
    print("-" * 48)
    print(f"{'合计':<16} {total_pt:>8} {total_ct:>12} {total_tt:>8}")
    print(f"\n总调用次数: {1 + meta_usage['calls']}")
else:
    print("LLM 未配置，无 API 调用。")

