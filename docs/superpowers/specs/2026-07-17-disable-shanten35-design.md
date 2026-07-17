# 暂停 shanten35、统一使用 shanten30 设计

## 目标

保留 `30`、`35` 两套向听惩罚参数及现有策略标记结构，但暂停线上 `35` 灰度分流，使所有测试服房间统一使用 `30`，用于后续约 300 局独立确认测试。

## 改动范围

- `_BASELINE_SHANTEN_PENALTY` 继续为 `30.0`。
- `_CANDIDATE_SHANTEN_PENALTY` 继续保留为 `35.0`，不删除。
- `_shanten_penalty_for_room()` 暂时不再按房间号奇偶分流，所有输入均返回 `30.0`。
- `strategy_variant()` 因实际惩罚统一为 30，线上日志统一记录 `two_ply_shanten30`。
- 不修改两层搜索、`continuation_weight=0.55`、决策时间预算、并发保护、合法兜底、动作逻辑和座位配置。

## 测试

1. 先修改现有房间奇偶测试，要求奇数、偶数和缺失房间号均使用 `30.0`，策略标记均为 `two_ply_shanten30`。
2. 在实现修改前运行该测试并确认它因奇数房间仍返回 35 而失败。
3. 最小修改选择函数后重新运行测试。
4. 运行九江 API 相关测试及完整九江测试集，确认没有其他行为回归。

## 回退方式

需要恢复灰度时，只需把 `_shanten_penalty_for_room()` 恢复为奇数房间返回 `_CANDIDATE_SHANTEN_PENALTY`、偶数或缺失房间返回 `_BASELINE_SHANTEN_PENALTY`；35常量和日志结构均仍保留。
