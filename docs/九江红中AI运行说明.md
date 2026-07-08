# 九江红中麻将 AI 运行说明

## 目录说明

- `jiujiang_ai/`：九江红中麻将 AI 主代码
- `jiujiang_ai/api.py`：对外决策入口，包含 `get_action(data)` 和 `round_end(data)`
- `jiujiang_ai/server.py`：标准库 HTTP 服务，提供 `/get_action` 和 `/round_end`
- `jiujiang_ai/hu.py`：胡牌判断，支持平胡、红中万能牌、四红中直接胡、可选七对
- `jiujiang_ai/ting.py`：听牌和真实有效进张判断
- `jiujiang_ai/stats.py`：单局统计、日志落盘、批量汇总、对打摘要
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

`/get_action` 返回：

```json
[action_type, action_card]
```

常用动作：

- `0, []`：过
- `2, [x, x, x]`：碰
- `3, [x, x, x, x]`：杠
- `4, []`：胡
- `7, [x]`：弃牌
- `8, []`：听牌

`/round_end` 会接收对局结束信息，并返回当前累计统计快照：

```json
{
  "status": "ok",
  "received": true,
  "data": {},
  "stats": {
    "total_rounds": 1,
    "win_count": 1,
    "self_draw_count": 0,
    "discard_win_count": 1,
    "wins_by_player": {"2": 1},
    "dianpao_by_player": {"1": 1},
    "total_score_by_player": {"0": 0.0, "1": -2.0, "2": 2.0, "3": 0.0}
  }
}
```

同时，`round_end(data)` 默认还会把每局结果追加写入：

```text
D:\MaJiang\logs\jiujiang_round_end.jsonl
```

格式是 JSONL，也就是一行一局，后续适合直接做批量统计。

## 对打结果汇总

如果你已经收集了一批 `/round_end` 的结果，可以把它们整理成一个 JSON 数组文件，然后执行：

```powershell
cd D:\MaJiang
python examples\jiujiang_match_report.py --input rounds.json --our-players 0,2
```

如果你直接使用了默认日志落盘，也可以不整理文件，直接读取默认日志：

```powershell
cd D:\MaJiang
python examples\jiujiang_match_report.py --use-default-log --our-players 0,2
```

其中：

- `--input`：局结果数组文件路径
- `--use-default-log`：直接读取 `logs\jiujiang_round_end.jsonl`
- `--our-players`：我方座位列表，例如 `0` 或 `0,2`

如果只是先看脚本输出格式，也可以直接跑内置样例：

```powershell
python examples\jiujiang_match_report.py --sample --our-players 0,2
```

输出会包含两部分：

- `overall`：整体统计
- `team`：站在我方座位视角的胜局、自摸、点炮、总分、局均分

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
- 出牌候选评估
- 杠牌收益第一版：若杠后向听变差，或同向听下手牌价值明显下降，则主动跳过杠牌
- 防守第一版：在进攻价值接近时，优先打场上已经出现过的相对安全牌
- 已听牌时保守跳过碰、杠
- 过碰限制：过碰同一张牌后，自己下次出牌前不再碰同一张牌
- HTTP `/get_action` 和 `/round_end` 接口
- `/round_end` 对局统计：总局数、胜局数、自摸次数、点炮胡次数、玩家胜局、点炮玩家、累计分数
- 批量对局汇总：支持把多局 `round_end` 结果汇总成整体统计和我方视角摘要
- 输入输出说明文档样例回放测试

## 当前边界

- 杠牌收益目前是第一版保守策略，还没有完整结算收益模型，也没有结合番型、对手状态做更细收益估计
- 防守目前只做到“已见弃牌越多越相对安全”的基础层，还没有做危险牌分级、他家听牌推断和更复杂的读牌逻辑
- `round_end` 已支持内存统计，但暂未落盘保存长期日志，服务重启后统计会清空
- API 层会过滤输入输出说明通用样例中的非九江牌；核心规则模块仍严格校验九江牌
