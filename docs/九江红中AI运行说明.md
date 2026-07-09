# 九江红中麻将 AI 运行说明

## 目录说明

- `jiujiang_ai/`：九江红中麻将 AI 主代码
- `jiujiang_ai/api.py`：对外决策入口，包含 `get_action(data)` 和 `round_end(data)`
- `jiujiang_ai/server.py`：标准库 HTTP 服务，提供 `/get_action` 和 `/round_end`
- `jiujiang_ai/hu.py`：胡牌判断，支持平胡、红中万能牌、四红中直接胡、七对开关
- `jiujiang_ai/ting.py`：听牌和真实有效进张判断
- `jiujiang_ai/evaluator.py`：弃牌评估与第一版防守逻辑
- `jiujiang_ai/settlement.py`：胡牌分、杠分、加买分、总分结算
- `jiujiang_ai/zama.py`：扎码分计算
- `jiujiang_ai/round_flow.py`：庄家轮转、黄庄/荒庄判断
- `jiujiang_ai/stats.py`：对局统计、日志落盘、批量汇总
- `examples/jiujiang_http_debug.py`：本地接口调试脚本
- `examples/jiujiang_match_report.py`：批量对局结果汇总脚本
- `tests/`：单元测试和接口回放测试

## 启动 HTTP 服务

在 PowerShell 中执行：

```powershell
cd D:\MaJiang
python -m jiujiang_ai.server
```

默认监听地址：

```text
http://127.0.0.1:8000/get_action
http://127.0.0.1:8000/round_end
```

## 本地直接调试

不启动 HTTP 服务，直接调用 Python 接口：

```powershell
cd D:\MaJiang
python examples\jiujiang_http_debug.py --direct
```

示例输出：

```json
{
  "request_action_cards": {
    "7": [[9], [24]]
  },
  "result": [7, [9]]
}
```

其中：

- `result[0]`：`action_type`
- `result[1]`：`action_card`

## HTTP 调试

先启动服务：

```powershell
cd D:\MaJiang
python -m jiujiang_ai.server
```

再开一个 PowerShell 窗口执行：

```powershell
cd D:\MaJiang
python examples\jiujiang_http_debug.py
```

如果服务地址不是默认端口，可以指定：

```powershell
python examples\jiujiang_http_debug.py --url http://127.0.0.1:9000/get_action
```

## 返回格式

### `/get_action`

返回：

```json
[action_type, action_card]
```

常用动作：

- `0, []`：过
- `2, [x, x, x]`：碰
- `3, [x, x, x, x]`：明杠
- `4, []`：胡
- `5, [x, x, x, x]`：暗杠
- `6, [x, x, x, x]`：补杠
- `7, [x]`：弃牌
- `8, []`：听牌

### `/round_end`

`/round_end` 会接收对局结束信息，并返回：

- 当前请求是否接收成功
- 当前累计统计 `stats`
- 当前局胡牌上下文 `win_context`
- 当前局完整结算结果 `settlement`

示例：

```json
{
  "status": "ok",
  "received": true,
  "data": {},
  "stats": {
    "total_rounds": 1,
    "win_count": 1,
    "self_draw_count": 1,
    "discard_win_count": 0,
    "win_type_count": {
      "zimo": 1
    },
    "wins_by_player": {
      "0": 1
    },
    "total_score_by_player": {}
  },
  "win_context": {
    "win_type": "zimo",
    "winners": [0],
    "dianpao_player": null,
    "is_multi_win": false
  },
  "settlement": {
    "winners": [0],
    "win_type": "zimo",
    "score_by_player": [9, -3, -3, -3],
    "components": {
      "hu": {
        "hu_score": 2.0,
        "score_by_player": [6.0, -2.0, -2.0, -2.0]
      },
      "buy": {
        "score_by_player": [0.0, 0.0, 0.0, 0.0]
      },
      "gang": {
        "score_by_player": [0, 0, 0, 0]
      },
      "zama": {
        "zama_score": 1,
        "score_by_player": [3.0, -1.0, -1.0, -1.0]
      }
    }
  }
}
```

## 当前已接入的结算模块

### 胡牌分

口径：

```text
胡牌分 = 胡型分 × 胡牌方式倍率 × 跑红中倍率
```

当前支持：

- 胡型分默认 `1`
- 自摸 `×2`
- 杠开 `×2`
- 点炮 `×1`
- 抢杠胡按当前规则实现并入结算链
- 跑红中翻倍：每出一个红中 `×2`

### 杠分

当前支持：

- 直杠：`3` 分，一家付
- 补杠：`1` 分，三家付
- 暗杠：`2` 分，三家付
- 荒庄荒杠：黄庄时杠分清零

### 加买分

当前支持：

- `enable_buy_score`
- `player_buy_scores / buy_scores / player_buy_score`
- 回退到统一房间 `buy_score`

计算口径：

```text
支付分数 = 自身买分 + 胡牌玩家买分
```

### 扎码分

当前支持：

- 胡牌后摸 `N` 张码牌
- 一码全中：摸到几奖几分
- 红中算 `10` 分
- 无红中胡牌多奖 `1` 码
- 四红中胡牌码分额外 `+4`
- 码跟底分：开启后按 `base_score` 再乘一次
- 自摸三家付码分
- 点炮一家付码分
- 一炮多响共用同一份码牌输入

### 总分

当前总分入口：

```python
calculate_total_score(data)
```

汇总口径：

```text
总分 = 胡牌分 + 加买分 + 杠分 + 码分
```

## 日志与对打汇总

`round_end(data)` 默认会把每局结果追加写入：

```text
D:\MaJiang\logs\jiujiang_round_end.jsonl
```

每行格式：

```json
{"timestamp": "...", "data": {...}, "stats": {...}}
```

### 对打结果汇总

如果你已经收集了一批 `/round_end` 结果，可以整理成一个 JSON 数组文件，然后执行：

```powershell
cd D:\MaJiang
python examples\jiujiang_match_report.py --input rounds.json --our-players 0,2
```

如果直接使用默认日志落盘，也可以执行：

```powershell
cd D:\MaJiang
python examples\jiujiang_match_report.py --use-default-log --our-players 0,2
```

其中：

- `--input`：对局结果数组文件路径
- `--use-default-log`：直接读取 `logs\jiujiang_round_end.jsonl`
- `--our-players`：我方座位列表，例如 `0` 或 `0,2`

如果只是先看脚本输出格式，也可以跑内置样例：

```powershell
python examples\jiujiang_match_report.py --sample --our-players 0,2
```

## 测试命令

九江相关全量测试：

```powershell
cd D:\MaJiang
python -m unittest discover -s tests -p "test_jiujiang*.py" -v
```

旧原型相关测试：

```powershell
python -m unittest tests.test_hand_split tests.test_search_tree -v
```

## 当前已支持

- 九江牌集：万、条、筒、红中
- 红中万能牌胡牌判断
- 红中不能碰、不能杠
- 不吃牌
- 四红中直接胡
- 平胡判断
- 七对房间选项判断
- 听牌和真实胡牌进张统计
- 弃牌候选评估
- 杠牌收益第一版
- 防守第一版
- 已听牌时保守跳过碰、杠
- 过碰限制
- HTTP `/get_action` 和 `/round_end` 接口
- `/round_end` 对局统计
- 胡牌方式识别
- 杠分 / 加买分 / 扎码分 / 总分结算
- 庄家轮转、黄庄/荒庄判断
- 输入输出说明文档样例回放测试

## 当前边界

- 杠牌收益目前仍是第一版保守策略，还没有结合更细的收益模型
- 防守逻辑目前只做到基础安全性排序，还没有他家听牌推断和危险牌分级
- 总分结算已经打通主链，但还没有进一步接入更复杂的房间特殊玩法扩展
- `stats` 是进程内累计统计，服务重启后会清空；长期结果依赖 JSONL 日志
