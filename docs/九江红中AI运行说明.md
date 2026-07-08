# 九江红中麻将 AI 运行说明

## 目录说明

- `jiujiang_ai/`：九江红中麻将 AI 主代码。
- `jiujiang_ai/api.py`：对外动作决策入口，包含 `get_action(data)` 和 `round_end(data)`。
- `jiujiang_ai/server.py`：标准库 HTTP 服务，提供 `/get_action` 和 `/round_end`。
- `jiujiang_ai/hu.py`：胡牌判断，支持平胡、红中万能牌、四红中直接胡、可选七对。
- `jiujiang_ai/ting.py`：听牌和真实有效进张判断。
- `examples/jiujiang_http_debug.py`：本地调试脚本。
- `tests/`：单元测试和接口回放测试。

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

## 本地直接调用

不启动 HTTP 服务，直接调用 Python 接口：

```powershell
cd D:\MaJiang
python examples\jiujiang_http_debug.py --direct
```

输出示例：

```json
{
  "request_action_cards": {
    "7": [[9], [24]]
  },
  "result": [7, [9]]
}
```

其中 `result[0]` 是 `action_type`，`result[1]` 是 `action_card`。

## HTTP 调试

先启动服务：

```powershell
cd D:\MaJiang
python -m jiujiang_ai.server
```

另开一个 PowerShell 窗口执行：

```powershell
cd D:\MaJiang
python examples\jiujiang_http_debug.py
```

如果服务地址不是默认端口，可以指定：

```powershell
python examples\jiujiang_http_debug.py --url http://127.0.0.1:9000/get_action
```

## 返回格式

`/get_action` 返回列表：

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

`/round_end` 当前返回确认信息：

```json
{
  "status": "ok",
  "received": true,
  "data": {}
}
```

## 跑测试

完整测试：

```powershell
cd D:\MaJiang
python -m unittest discover -s tests -v
```

只跑九江相关测试：

```powershell
python -m unittest tests.test_jiujiang_api tests.test_jiujiang_hu tests.test_jiujiang_ting -v
```

## 当前已支持

- 九江牌集：万、条、筒、红中。
- 红中万能牌胡牌判断。
- 红中不能碰、不能杠。
- 不吃牌。
- 四红中直接胡。
- 平胡判断。
- 七对房间选项判断。
- 听牌和真实胡牌进张统计。
- 出牌候选评估。
- 已听牌时保守跳过碰/杠。
- 过碰限制：过碰同一张牌后，自己下次出牌前不再碰同一张牌。
- HTTP `/get_action` 和 `/round_end` 接口。
- 输入输出说明文档样例回放测试。

## 当前边界

- 杠牌收益还只是保守策略，没有完整结算收益模型。
- 危险牌、防守、他家听牌推断还没做。
- `round_end` 目前只确认接收，暂未统计长期胜率或对局日志。
- API 层会过滤输入输出说明通用样例中的非九江牌；核心规则模块仍严格校验九江牌。
