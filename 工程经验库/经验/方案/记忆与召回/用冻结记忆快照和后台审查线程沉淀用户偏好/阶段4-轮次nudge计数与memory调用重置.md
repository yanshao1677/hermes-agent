---
title: 阶段4-轮次nudge计数与memory调用重置
level: atomic
parent: 用冻结记忆快照和后台审查线程沉淀用户偏好
status: draft
tags:
  - memory
  - nudge
created_at: 2026-07-02
updated_at: 2026-07-02
confidence: high
related:
  - 阶段7-主响应交付后后台审查记忆和技能.md
source_repo: hermes-agent
source_commit: ce1dc9f77865c2c92c9e39f45fcf9c3ca46ecfdc
source_paths:
  - run_agent.py
---

# 阶段4：轮次 nudge 计数与 memory 调用重置

## 触发条件

每个用户回合进入 `run_conversation()`；或在工具批次中解析到 `memory` 工具调用。

## 输入字段

| 字段 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `memory_nudge_interval` | int | 10 | 配置 `memory.nudge_interval`；0=禁用 |
| `turns_since_memory` | int | 0 | 实例计数器 |
| `valid_tool_names` | set | — | 须含 `"memory"` |
| `_memory_store` | MemoryStore | — | 须非空 |

## 判定规则

**回合开始（每个 user turn）**

1. 若 `memory_nudge_interval > 0` 且 `"memory" in valid_tool_names` 且 `_memory_store` 存在：
   - `turns_since_memory += 1`
   - 若 `turns_since_memory >= memory_nudge_interval`：
     - `_should_review_memory = True`
     - `turns_since_memory = 0`（预清零，非审查完成时清零）
2. 否则：不递增，不设置审查标志。

**工具执行（memory 被调用）**

1. 在并发或串行工具路径，当 `function_name == "memory"`：
   - `turns_since_memory = 0`（模型已主动管理记忆，跳过本轮 nudge）。

**与技能 nudge 独立**

- `iters_since_skill` 在工具循环按迭代累加，回合结束判定；与 memory nudge 可并行触发 combined 审查。

## 状态读写位置

- 读写：实例字段 `turns_since_memory`、回合局部 `_should_review_memory`。
- 不持久化到 DB。

## 正常路径

1. Turn 1-9：计数递增，无审查。
2. Turn 10：设置 `_should_review_memory`，计数归零。
3. 主循环结束交付响应后阶段 7 读取标志 spawn 审查。
4. Turn 5 中模型调用 memory → 计数归零，Turn 10 审查顺延。

## 分支路径

- `memory_nudge_interval=0` → 永不自动审查。
- 无 memory 工具或无 store → 不递增。
- 审查 spawn 失败 → 不影响主响应；计数已在回合开始处理完毕。

## 失败处理

无阻断失败；计数器为 best-effort 启发式，非严格调度。

## 幂等性 / 一致性约束

`_should_review_memory` 为单回合布尔标志，不跨回合持久。同一回合 memory 工具调用与阈值触发可能同时发生：memory 调用重置计数，但已设置的 `_should_review_memory` 不会被清除（证据不足：是否在 memory 调用后取消已设标志——源码在回合开始设标志，工具循环在之后执行，若同轮既达阈值又调 memory，审查仍可能触发）。

## 代码骨架

```python
def on_turn_start(agent):
    agent.should_review_memory = False
    if agent.memory_nudge_interval > 0 and agent.has_memory_tool:
        agent.turns_since_memory += 1
        if agent.turns_since_memory >= agent.memory_nudge_interval:
            agent.should_review_memory = True
            agent.turns_since_memory = 0

def on_memory_tool(agent):
    agent.turns_since_memory = 0
```

## 最小验证清单

- interval=10 时第 10 个 user turn 设置审查标志。
- 中间 turn 调用 memory 后计数从 0 重新累加。
- interval=0 时不递增。

## 来源证据（附录，不进正文）

- `run_agent.py:1086-1088`：默认值。
- `run_agent.py:7506-7514`：回合开始递增与触发。
- `run_agent.py:6680-6681`：并发工具路径重置。
- `run_agent.py:6883`：串行工具路径重置（grep 第二处）。
- `run_agent.py:7476`：注释说明计数不在 run 开头全局重置。
