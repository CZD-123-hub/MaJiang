# 九江红中麻将 AI 实现计划

> **给执行 agent 的要求：** 必须使用 `superpowers:subagent-driven-development`（推荐）或 `superpowers:executing-plans` 按任务执行。所有步骤使用 checkbox（`- [ ]`）跟踪。

**目标：** 新增一套独立的九江红中麻将 AI，实现九江规则校验、红中万能牌平胡评估、动作选择，以及 `get_action` / `round_end` 可调用入口。

**架构：** 新建独立 `jiujiang_ai` 包，不直接修改 `v5_rh`。模块按职责拆分为：牌集工具、动作规则、手牌分析、弃牌评估、API 编排。开发过程采用 TDD：先写失败测试，再写最小实现，再运行测试验证。

**技术栈：** Python 3.12、标准库、`unittest`、v5 风格牌编码。

---

## 文件结构

- 新建：`D:/MaJiang/jiujiang_ai/__init__.py`
  - 导出公共 API 和核心常量。
- 新建：`D:/MaJiang/jiujiang_ai/tiles.py`
  - 负责九江红中牌常量、合法性校验、格式化、剩余牌数量统计。
- 新建：`D:/MaJiang/jiujiang_ai/rules.py`
  - 负责动作常量和九江规则校验，例如不可吃、红中不能碰杠。
- 新建：`D:/MaJiang/jiujiang_ai/hand_split.py`
  - 负责平胡手牌分析、普通数牌拆分、红中万能牌向听数修正。
- 新建：`D:/MaJiang/jiujiang_ai/evaluator.py`
  - 负责候选弃牌评分，以及操作前后手牌价值评估。
- 新建：`D:/MaJiang/jiujiang_ai/api.py`
  - 负责 `get_action(data)` 和 `round_end(data)`。
- 新建：`D:/MaJiang/tests/test_jiujiang_tiles.py`
- 新建：`D:/MaJiang/tests/test_jiujiang_rules.py`
- 新建：`D:/MaJiang/tests/test_jiujiang_hand_split.py`
- 新建：`D:/MaJiang/tests/test_jiujiang_evaluator.py`
- 新建：`D:/MaJiang/tests/test_jiujiang_api.py`

---

### 任务 1：九江红中牌集

**文件：**
- 新建：`D:/MaJiang/tests/test_jiujiang_tiles.py`
- 新建：`D:/MaJiang/jiujiang_ai/__init__.py`
- 新建：`D:/MaJiang/jiujiang_ai/tiles.py`

- [ ] **步骤 1：编写失败测试**

创建 `D:/MaJiang/tests/test_jiujiang_tiles.py`：

```python
import unittest

from jiujiang_ai.tiles import (
    HONGZHONG,
    JIUJIANG_TILE_CODES,
    format_tiles,
    remaining_tile_counts,
    validate_hand,
)


class JiujiangTileTests(unittest.TestCase):
    def test_accepts_suited_tiles_and_hongzhong(self):
        hand = [0x01, 0x02, 0x03, 0x11, 0x12, 0x13, 0x21, 0x22, 0x23, HONGZHONG]

        validate_hand(hand)

        self.assertIn(HONGZHONG, JIUJIANG_TILE_CODES)
        self.assertEqual(format_tiles([0x01, 0x19, 0x29, HONGZHONG]), ["1W", "9T", "9B", "RED"])

    def test_rejects_non_jiujiang_honor_tiles(self):
        with self.assertRaises(ValueError):
            validate_hand([0x31])

        with self.assertRaises(ValueError):
            validate_hand([0x37])

    def test_rejects_more_than_four_of_same_tile(self):
        with self.assertRaises(ValueError):
            validate_hand([0x01, 0x01, 0x01, 0x01, 0x01])

    def test_remaining_tile_counts_uses_four_copies(self):
        counts = remaining_tile_counts([0x01, 0x01, HONGZHONG])

        self.assertEqual(counts[0x01], 2)
        self.assertEqual(counts[HONGZHONG], 3)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **步骤 2：运行测试确认失败**

运行：

```powershell
python -m unittest tests.test_jiujiang_tiles -v
```

预期：失败，报 `ModuleNotFoundError: No module named 'jiujiang_ai'`。

- [ ] **步骤 3：实现牌集常量与校验**

创建 `D:/MaJiang/jiujiang_ai/__init__.py`：

```python
"""Jiujiang Hongzhong Mahjong AI package."""

from .api import get_action, round_end
from .tiles import HONGZHONG

__all__ = ["HONGZHONG", "get_action", "round_end"]
```

创建 `D:/MaJiang/jiujiang_ai/tiles.py`：

```python
from __future__ import annotations

from collections import Counter

WAN_CODES = tuple(range(0x01, 0x0A))
TIAO_CODES = tuple(range(0x11, 0x1A))
TONG_CODES = tuple(range(0x21, 0x2A))
HONGZHONG = 0x35

SUITED_TILE_CODES = WAN_CODES + TIAO_CODES + TONG_CODES
JIUJIANG_TILE_CODES = SUITED_TILE_CODES + (HONGZHONG,)
JIUJIANG_TILE_SET = set(JIUJIANG_TILE_CODES)


def validate_tile(tile: int) -> None:
    if tile not in JIUJIANG_TILE_SET:
        raise ValueError(f"invalid Jiujiang tile: {tile!r}")


def validate_hand(tiles: list[int] | tuple[int, ...]) -> None:
    invalid = [tile for tile in tiles if tile not in JIUJIANG_TILE_SET]
    if invalid:
        raise ValueError(f"hand contains invalid Jiujiang tiles: {invalid!r}")
    over_limit = [tile for tile, count in Counter(tiles).items() if count > 4]
    if over_limit:
        raise ValueError(f"tile count exceeds four copies: {over_limit!r}")


def is_hongzhong(tile: int) -> bool:
    return tile == HONGZHONG


def is_suited(tile: int) -> bool:
    return tile in set(SUITED_TILE_CODES)


def tile_rank(tile: int) -> int:
    validate_tile(tile)
    return tile & 0x0F


def tile_suit(tile: int) -> str:
    validate_tile(tile)
    if tile == HONGZHONG:
        return "Z"
    prefix = tile & 0xF0
    if prefix == 0x00:
        return "W"
    if prefix == 0x10:
        return "T"
    return "B"


def can_start_sequence(tile: int) -> bool:
    return is_suited(tile) and tile_rank(tile) <= 7


def tile_name(tile: int) -> str:
    validate_tile(tile)
    if tile == HONGZHONG:
        return "RED"
    return f"{tile_rank(tile)}{tile_suit(tile)}"


def format_tiles(tiles: list[int] | tuple[int, ...]) -> list[str]:
    return [tile_name(tile) for tile in tiles]


def remaining_tile_counts(known_tiles: list[int] | tuple[int, ...]) -> dict[int, int]:
    validate_hand(known_tiles)
    known = Counter(known_tiles)
    return {tile: 4 - known[tile] for tile in JIUJIANG_TILE_CODES}
```

- [ ] **步骤 4：添加临时 API 桩，保证包可以导入**

创建 `D:/MaJiang/jiujiang_ai/api.py`：

```python
from __future__ import annotations


def get_action(data: dict) -> tuple[int, list[int]]:
    raise NotImplementedError("get_action is implemented in Task 6")


def round_end(data: dict) -> dict[str, object]:
    raise NotImplementedError("round_end is implemented in Task 6")
```

- [ ] **步骤 5：运行测试确认通过**

运行：

```powershell
python -m unittest tests.test_jiujiang_tiles -v
```

预期：通过。

---

### 任务 2：九江红中动作规则

**文件：**
- 新建：`D:/MaJiang/tests/test_jiujiang_rules.py`
- 新建：`D:/MaJiang/jiujiang_ai/rules.py`

- [ ] **步骤 1：编写失败测试**

创建 `D:/MaJiang/tests/test_jiujiang_rules.py`：

```python
import unittest

from jiujiang_ai.rules import (
    ACTION_CHI,
    ACTION_GANG,
    ACTION_HU,
    ACTION_PASS,
    ACTION_PENG,
    ACTION_DISCARD,
    is_legal_operation,
    normalize_action_cards,
)
from jiujiang_ai.tiles import HONGZHONG


class JiujiangRuleTests(unittest.TestCase):
    def test_action_constants_match_harness_values(self):
        self.assertEqual(ACTION_PASS, 0)
        self.assertEqual(ACTION_CHI, 1)
        self.assertEqual(ACTION_PENG, 2)
        self.assertEqual(ACTION_GANG, 3)
        self.assertEqual(ACTION_HU, 4)
        self.assertEqual(ACTION_DISCARD, 7)

    def test_chi_is_never_legal(self):
        self.assertFalse(is_legal_operation(ACTION_CHI, [0x01, 0x02, 0x03]))

    def test_hongzhong_cannot_peng_or_gang(self):
        self.assertFalse(is_legal_operation(ACTION_PENG, [HONGZHONG, HONGZHONG, HONGZHONG]))
        self.assertFalse(is_legal_operation(ACTION_GANG, [HONGZHONG, HONGZHONG, HONGZHONG, HONGZHONG]))

    def test_non_hongzhong_peng_and_gang_are_legal(self):
        self.assertTrue(is_legal_operation(ACTION_PENG, [0x02, 0x02, 0x02]))
        self.assertTrue(is_legal_operation(ACTION_GANG, [0x02, 0x02, 0x02, 0x02]))

    def test_normalize_action_cards_accepts_string_keys(self):
        normalized = normalize_action_cards({"7": [[0x01]], "4": []})

        self.assertIn(ACTION_DISCARD, normalized)
        self.assertIn(ACTION_HU, normalized)
        self.assertEqual(normalized[ACTION_DISCARD], [[0x01]])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **步骤 2：运行测试确认失败**

运行：

```powershell
python -m unittest tests.test_jiujiang_rules -v
```

预期：失败，报 `ModuleNotFoundError`，因为 `jiujiang_ai.rules` 尚未创建。

- [ ] **步骤 3：实现动作常量与合法性判断**

创建 `D:/MaJiang/jiujiang_ai/rules.py`：

```python
from __future__ import annotations

from .tiles import HONGZHONG

ACTION_PASS = 0
ACTION_CHI = 1
ACTION_PENG = 2
ACTION_GANG = 3
ACTION_HU = 4
ACTION_ANGANG = 5
ACTION_BUGANG = 6
ACTION_DISCARD = 7
ACTION_TING = 8

GANG_ACTIONS = {ACTION_GANG, ACTION_ANGANG, ACTION_BUGANG}


def normalize_action_cards(action_cards: dict[str | int, list[list[int]]]) -> dict[int, list[list[int]]]:
    normalized: dict[int, list[list[int]]] = {}
    for action_type, cards in action_cards.items():
        normalized[int(action_type)] = cards
    return normalized


def is_legal_operation(action_type: int, cards: list[int]) -> bool:
    if action_type == ACTION_CHI:
        return False
    if action_type == ACTION_PENG:
        return bool(cards) and HONGZHONG not in cards
    if action_type in GANG_ACTIONS:
        return bool(cards) and HONGZHONG not in cards
    if action_type in {ACTION_PASS, ACTION_HU, ACTION_DISCARD, ACTION_TING}:
        return True
    return False
```

- [ ] **步骤 4：运行测试确认通过**

运行：

```powershell
python -m unittest tests.test_jiujiang_rules -v
```

预期：通过。

---

### 任务 3：红中万能牌手牌分析

**文件：**
- 新建：`D:/MaJiang/tests/test_jiujiang_hand_split.py`
- 新建：`D:/MaJiang/jiujiang_ai/hand_split.py`

- [ ] **步骤 1：编写失败测试**

创建 `D:/MaJiang/tests/test_jiujiang_hand_split.py`：

```python
import unittest

from jiujiang_ai.hand_split import analyze_hand, is_four_hongzhong
from jiujiang_ai.tiles import HONGZHONG


class JiujiangHandSplitTests(unittest.TestCase):
    def test_complete_pinghu_without_hongzhong(self):
        hand = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x11, 0x12, 0x13, 0x21, 0x22, 0x23, 0x09, 0x09]

        result = analyze_hand(hand)

        self.assertEqual(result.shanten, 0)
        self.assertEqual(result.hongzhong_used, 0)

    def test_one_hongzhong_completes_pair(self):
        hand = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x11, 0x12, 0x13, 0x21, 0x22, 0x23, 0x09, HONGZHONG]

        result = analyze_hand(hand)

        self.assertEqual(result.shanten, 0)
        self.assertGreaterEqual(result.hongzhong_used, 1)

    def test_hongzhong_improves_incomplete_hand(self):
        no_red = analyze_hand([0x01, 0x02, 0x03, 0x04, 0x05, 0x11, 0x12, 0x21, 0x22, 0x09, 0x09, 0x18, 0x19])
        with_red = analyze_hand([0x01, 0x02, 0x03, 0x04, 0x05, 0x11, 0x12, 0x21, 0x22, 0x09, 0x09, 0x18, HONGZHONG])

        self.assertLessEqual(with_red.shanten, no_red.shanten)

    def test_four_hongzhong_detection(self):
        self.assertTrue(is_four_hongzhong([HONGZHONG, HONGZHONG, HONGZHONG, HONGZHONG]))
        self.assertFalse(is_four_hongzhong([HONGZHONG, HONGZHONG, HONGZHONG, 0x01]))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **步骤 2：运行测试确认失败**

运行：

```powershell
python -m unittest tests.test_jiujiang_hand_split -v
```

预期：失败，报 `ModuleNotFoundError`，因为 `jiujiang_ai.hand_split` 尚未创建。

- [ ] **步骤 3：实现红中平胡分析**

创建 `D:/MaJiang/jiujiang_ai/hand_split.py`：

```python
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache

from .tiles import HONGZHONG, SUITED_TILE_CODES, can_start_sequence, is_suited, tile_rank, validate_hand


@dataclass(frozen=True)
class HandAnalysis:
    shanten: int
    melds: int
    pairs: int
    taatsu: int
    leftovers: int
    hongzhong_count: int
    hongzhong_used: int


@dataclass(frozen=True)
class _Partial:
    melds: int = 0
    pairs: int = 0
    taatsu: int = 0
    leftovers: int = 0


def is_four_hongzhong(tiles: list[int] | tuple[int, ...]) -> bool:
    return list(tiles).count(HONGZHONG) >= 4


def analyze_hand(hand: list[int] | tuple[int, ...]) -> HandAnalysis:
    validate_hand(hand)
    hongzhong_count = list(hand).count(HONGZHONG)
    ordinary = tuple(sorted(tile for tile in hand if tile != HONGZHONG))
    partials = _split_counts(_counts_tuple(ordinary))
    best = min((_apply_hongzhong(partial, hongzhong_count) for partial in partials), key=_analysis_sort_key)
    return best


def _counts_tuple(tiles: tuple[int, ...]) -> tuple[int, ...]:
    counts = Counter(tiles)
    return tuple(counts[tile] for tile in SUITED_TILE_CODES)


@lru_cache(maxsize=None)
def _split_counts(counts: tuple[int, ...]) -> tuple[_Partial, ...]:
    try:
        index = next(i for i, count in enumerate(counts) if count)
    except StopIteration:
        return (_Partial(),)

    tile = SUITED_TILE_CODES[index]
    results: list[_Partial] = []

    def add_branch(new_counts: tuple[int, ...], melds: int = 0, pairs: int = 0, taatsu: int = 0, leftovers: int = 0) -> None:
        for child in _split_counts(new_counts):
            results.append(
                _Partial(
                    melds=child.melds + melds,
                    pairs=child.pairs + pairs,
                    taatsu=child.taatsu + taatsu,
                    leftovers=child.leftovers + leftovers,
                )
            )

    if counts[index] >= 3:
        add_branch(_remove_tiles(counts, (tile, tile, tile)), melds=1)

    if can_start_sequence(tile):
        seq = (tile, tile + 1, tile + 2)
        if _has_tiles(counts, seq):
            add_branch(_remove_tiles(counts, seq), melds=1)

    if counts[index] >= 2:
        add_branch(_remove_tiles(counts, (tile, tile)), pairs=1)

    if is_suited(tile):
        if tile_rank(tile) <= 8 and _has_tiles(counts, (tile, tile + 1)):
            add_branch(_remove_tiles(counts, (tile, tile + 1)), taatsu=1)
        if tile_rank(tile) <= 7 and _has_tiles(counts, (tile, tile + 2)):
            add_branch(_remove_tiles(counts, (tile, tile + 2)), taatsu=1)

    add_branch(_remove_tiles(counts, (tile,)), leftovers=1)
    return tuple(results)


def _has_tiles(counts: tuple[int, ...], tiles: tuple[int, ...]) -> bool:
    needed = Counter(tiles)
    return all(counts[SUITED_TILE_CODES.index(tile)] >= amount for tile, amount in needed.items())


def _remove_tiles(counts: tuple[int, ...], tiles: tuple[int, ...]) -> tuple[int, ...]:
    new_counts = list(counts)
    for tile in tiles:
        new_counts[SUITED_TILE_CODES.index(tile)] -= 1
    return tuple(new_counts)


def _apply_hongzhong(partial: _Partial, hongzhong_count: int) -> HandAnalysis:
    best: HandAnalysis | None = None
    for used_for_pair in range(hongzhong_count + 1):
        for used_for_melds in range(hongzhong_count - used_for_pair + 1):
            pairs = partial.pairs
            melds = partial.melds
            taatsu = partial.taatsu
            leftovers = partial.leftovers
            used = 0

            if used_for_pair:
                if pairs == 0:
                    pairs = 1
                    leftovers = max(0, leftovers - 1) if used_for_pair == 1 else leftovers
                    used += min(2, used_for_pair)

            promoted_taatsu = min(taatsu, used_for_melds)
            melds += promoted_taatsu
            taatsu -= promoted_taatsu
            used += promoted_taatsu

            remaining_red = hongzhong_count - used
            promoted_leftovers = min(leftovers, remaining_red // 2)
            melds += promoted_leftovers
            leftovers -= promoted_leftovers
            used += promoted_leftovers * 2

            analysis = _calculate_analysis(melds, pairs, taatsu, leftovers, hongzhong_count, used)
            if best is None or _analysis_sort_key(analysis) < _analysis_sort_key(best):
                best = analysis

    assert best is not None
    return best


def _calculate_analysis(
    melds: int,
    pairs: int,
    taatsu: int,
    leftovers: int,
    hongzhong_count: int,
    hongzhong_used: int,
) -> HandAnalysis:
    capped_melds = min(4, melds)
    has_pair = pairs > 0
    extra_pairs_as_taatsu = max(0, pairs - (1 if has_pair else 0))
    incomplete = extra_pairs_as_taatsu + taatsu
    useful_incomplete = min(incomplete, max(0, 4 - capped_melds))
    shanten = 9 - 2 * capped_melds - useful_incomplete - (1 if has_pair else 0)
    return HandAnalysis(
        shanten=max(0, shanten),
        melds=capped_melds,
        pairs=pairs,
        taatsu=taatsu,
        leftovers=leftovers,
        hongzhong_count=hongzhong_count,
        hongzhong_used=hongzhong_used,
    )


def _analysis_sort_key(analysis: HandAnalysis) -> tuple[int, int, int, int, int]:
    return (
        analysis.shanten,
        -analysis.melds,
        -(analysis.pairs + analysis.taatsu),
        analysis.leftovers,
        analysis.hongzhong_used,
    )
```

- [ ] **步骤 4：运行测试确认通过**

运行：

```powershell
python -m unittest tests.test_jiujiang_hand_split -v
```

预期：通过。

---

### 任务 4：弃牌评估器

**文件：**
- 新建：`D:/MaJiang/tests/test_jiujiang_evaluator.py`
- 新建：`D:/MaJiang/jiujiang_ai/evaluator.py`

- [ ] **步骤 1：编写失败测试**

创建 `D:/MaJiang/tests/test_jiujiang_evaluator.py`：

```python
import unittest

from jiujiang_ai.evaluator import choose_discard, score_discards


class JiujiangEvaluatorTests(unittest.TestCase):
    def test_prefers_isolated_tile_over_breaking_meld(self):
        hand = [0x01, 0x02, 0x03, 0x11, 0x12, 0x13, 0x21, 0x22, 0x23, 0x05, 0x05, 0x05, 0x08, 0x09]
        candidates = [[0x01], [0x08]]

        decision = choose_discard(hand, candidates)

        self.assertEqual(decision.discard, 0x08)

    def test_scores_only_provided_candidates(self):
        hand = [0x01, 0x02, 0x03, 0x11, 0x12, 0x13, 0x21, 0x22, 0x23, 0x05, 0x05, 0x05, 0x08, 0x09]
        scores = score_discards(hand, [[0x01], [0x08]])

        self.assertEqual(set(scores), {0x01, 0x08})

    def test_prefers_better_shanten_when_obvious(self):
        hand = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x11, 0x12, 0x13, 0x21, 0x22, 0x23, 0x09, 0x18]
        decision = choose_discard(hand, [[0x09], [0x18]])

        self.assertIn(decision.discard, {0x09, 0x18})


if __name__ == "__main__":
    unittest.main()
```

- [ ] **步骤 2：运行测试确认失败**

运行：

```powershell
python -m unittest tests.test_jiujiang_evaluator -v
```

预期：失败，报 `ModuleNotFoundError`，因为 `jiujiang_ai.evaluator` 尚未创建。

- [ ] **步骤 3：实现弃牌评分**

创建 `D:/MaJiang/jiujiang_ai/evaluator.py`：

```python
from __future__ import annotations

from dataclasses import dataclass

from .hand_split import analyze_hand
from .tiles import JIUJIANG_TILE_CODES, remaining_tile_counts, validate_hand


@dataclass(frozen=True)
class DiscardDecision:
    discard: int
    score: float
    shanten_after_discard: int
    effective_count: int


def score_discards(hand: list[int], candidate_cards: list[list[int]]) -> dict[int, DiscardDecision]:
    validate_hand(hand)
    scores: dict[int, DiscardDecision] = {}
    for card_group in candidate_cards:
        if not card_group:
            continue
        discard = card_group[0]
        if discard not in hand:
            continue
        after = list(hand)
        after.remove(discard)
        analysis = analyze_hand(after)
        effective_count = _effective_draw_count(after, analysis.shanten)
        score = _score_analysis(analysis, effective_count)
        scores[discard] = DiscardDecision(
            discard=discard,
            score=score,
            shanten_after_discard=analysis.shanten,
            effective_count=effective_count,
        )
    return scores


def choose_discard(hand: list[int], candidate_cards: list[list[int]]) -> DiscardDecision:
    scores = score_discards(hand, candidate_cards)
    if not scores:
        raise ValueError("no valid discard candidates")
    return max(scores.values(), key=lambda decision: (decision.score, -decision.shanten_after_discard, -decision.discard))


def hand_value(hand: list[int]) -> float:
    analysis = analyze_hand(hand)
    return _score_analysis(analysis, _effective_draw_count(hand, analysis.shanten))


def _effective_draw_count(hand: list[int], current_shanten: int) -> int:
    remaining = remaining_tile_counts(hand)
    count = 0
    for tile in JIUJIANG_TILE_CODES:
        if remaining[tile] <= 0:
            continue
        next_hand = sorted([*hand, tile])
        if analyze_hand(next_hand).shanten < current_shanten:
            count += remaining[tile]
    return count


def _score_analysis(analysis, effective_count: int) -> float:
    return (
        100
        - 30 * analysis.shanten
        + 4 * analysis.melds
        + 1.5 * analysis.taatsu
        + 1.0 * analysis.pairs
        + 0.8 * effective_count
        - 0.5 * analysis.leftovers
        - 0.3 * analysis.hongzhong_used
    )
```

- [ ] **步骤 4：运行测试确认通过**

运行：

```powershell
python -m unittest tests.test_jiujiang_evaluator -v
```

预期：通过。

---

### 任务 5：公共动作 API

**文件：**
- 新建：`D:/MaJiang/tests/test_jiujiang_api.py`
- 修改：`D:/MaJiang/jiujiang_ai/api.py`

- [ ] **步骤 1：编写失败测试**

创建 `D:/MaJiang/tests/test_jiujiang_api.py`：

```python
import unittest

from jiujiang_ai.api import get_action, round_end
from jiujiang_ai.rules import ACTION_GANG, ACTION_HU, ACTION_PASS, ACTION_PENG, ACTION_DISCARD
from jiujiang_ai.tiles import HONGZHONG


class JiujiangApiTests(unittest.TestCase):
    def test_hu_has_priority(self):
        data = {"action_cards": {"4": [], "7": [[0x01]]}, "player_hand_cards": [[0x01] * 2, [], [], []], "acting_do_player_position": 0}

        action_type, action_card = get_action(data)

        self.assertEqual(action_type, ACTION_HU)
        self.assertEqual(action_card, [])

    def test_discards_from_candidates(self):
        hand = [0x01, 0x02, 0x03, 0x11, 0x12, 0x13, 0x21, 0x22, 0x23, 0x05, 0x05, 0x05, 0x08, 0x09]
        data = {"action_cards": {"7": [[0x01], [0x08]]}, "player_hand_cards": [hand, [], [], []], "acting_do_player_position": 0}

        action_type, action_card = get_action(data)

        self.assertEqual(action_type, ACTION_DISCARD)
        self.assertEqual(action_card, [0x08])

    def test_rejects_chi_and_hongzhong_peng_then_passes(self):
        data = {"action_cards": {"1": [[0x01, 0x02, 0x03]], "2": [[HONGZHONG, HONGZHONG, HONGZHONG]], "0": []}}

        action_type, action_card = get_action(data)

        self.assertEqual(action_type, ACTION_PASS)
        self.assertEqual(action_card, [])

    def test_selects_non_hongzhong_gang(self):
        data = {"action_cards": {"3": [[0x02, 0x02, 0x02, 0x02]]}}

        action_type, action_card = get_action(data)

        self.assertEqual(action_type, ACTION_GANG)
        self.assertEqual(action_card, [0x02, 0x02, 0x02, 0x02])

    def test_round_end_acknowledges_payload(self):
        result = round_end({"winner": 0})

        self.assertEqual(result["status"], "ok")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **步骤 2：运行测试确认失败**

运行：

```powershell
python -m unittest tests.test_jiujiang_api -v
```

预期：失败，因为 `api.py` 仍然抛出 `NotImplementedError`。

- [ ] **步骤 3：实现 API 编排逻辑**

修改 `D:/MaJiang/jiujiang_ai/api.py`：

```python
from __future__ import annotations

from .evaluator import choose_discard, hand_value
from .rules import (
    ACTION_GANG,
    ACTION_HU,
    ACTION_PASS,
    ACTION_PENG,
    ACTION_DISCARD,
    GANG_ACTIONS,
    is_legal_operation,
    normalize_action_cards,
)


def get_action(data: dict) -> tuple[int, list[int]]:
    action_cards = normalize_action_cards(data.get("action_cards", {}))

    if ACTION_HU in action_cards:
        return ACTION_HU, []

    for gang_action in sorted(GANG_ACTIONS):
        for cards in action_cards.get(gang_action, []):
            if is_legal_operation(gang_action, cards):
                return ACTION_GANG, cards

    hand = _acting_hand(data)
    best_peng = _best_peng(action_cards, hand)
    if best_peng is not None:
        return ACTION_PENG, best_peng

    if ACTION_DISCARD in action_cards and hand:
        decision = choose_discard(hand, action_cards[ACTION_DISCARD])
        return ACTION_DISCARD, [decision.discard]

    return ACTION_PASS, []


def round_end(data: dict) -> dict[str, object]:
    return {"status": "ok", "received": True, "data": data}


def _acting_hand(data: dict) -> list[int]:
    hands = data.get("player_hand_cards") or []
    position = int(data.get("acting_do_player_position", 0))
    if 0 <= position < len(hands):
        return list(hands[position])
    return []


def _best_peng(action_cards: dict[int, list[list[int]]], hand: list[int]) -> list[int] | None:
    best_cards: list[int] | None = None
    best_value = hand_value(hand) if hand else 0
    for cards in action_cards.get(ACTION_PENG, []):
        if not is_legal_operation(ACTION_PENG, cards):
            continue
        simulated = list(hand)
        for tile in cards[:2]:
            if tile in simulated:
                simulated.remove(tile)
        value = hand_value(simulated) if simulated else 0
        if value > best_value:
            best_value = value
            best_cards = cards
    return best_cards
```

- [ ] **步骤 4：运行测试确认通过**

运行：

```powershell
python -m unittest tests.test_jiujiang_api -v
```

预期：通过。

---

### 任务 6：完整验证

**文件：**
- 如有必要，修改任务 1-5 中创建的文件。

- [ ] **步骤 1：运行完整测试套件**

运行：

```powershell
python -m unittest discover -s tests -v
```

预期：全部已有测试和新测试通过。

- [ ] **步骤 2：手动运行 `get_action` 样例**

PowerShell 下运行：

```powershell
@'
from jiujiang_ai.api import get_action, round_end
from jiujiang_ai.tiles import HONGZHONG

print(get_action({"action_cards": {"4": []}}))
print(get_action({"action_cards": {"3": [[2, 2, 2, 2]]}}))
print(get_action({"action_cards": {"2": [[HONGZHONG, HONGZHONG, HONGZHONG]], "0": []}}))
print(round_end({"winner": 0}))
'@ | python -
```

预期输出包含：

```text
(4, [])
(3, [2, 2, 2, 2])
(0, [])
{'status': 'ok', 'received': True, 'data': {'winner': 0}}
```

- [ ] **步骤 3：复查 OpenSpec 覆盖关系**

打开：

```text
D:/MaJiang/openspec/changes/add-jiujiang-red-ai/tasks.md
D:/MaJiang/openspec/changes/add-jiujiang-red-ai/specs/jiujiang-red-ai/spec.md
```

确认每条需求都有对应测试或实现路径：

- 九江红中牌集校验：任务 1。
- 红中万能牌手牌分析：任务 3。
- 九江动作合法性：任务 2、任务 5。
- 动作选择输出：任务 5。
- 弃牌评估：任务 4。
- 操作评估：任务 5。
- 对局结束处理：任务 5。

- [ ] **步骤 4：检查 git 状态**

运行：

```powershell
git status --short
```

预期：可以看到新增的 `jiujiang_ai/`、测试、OpenSpec 工件和计划文件。除非用户要求，不要暂存或提交。

---

## 自检

需求覆盖：

- `九江红中牌集校验`：任务 1 覆盖。
- `红中万能牌手牌分析`：任务 3 覆盖。
- `九江动作合法性`：任务 2 和任务 5 覆盖。
- `动作选择输出`：任务 5 覆盖。
- `弃牌评估`：任务 4 覆盖。
- `操作评估`：任务 5 覆盖。
- `对局结束处理`：任务 5 覆盖。

占位内容检查：

- 没有保留占位符或未说明的实现步骤。

类型一致性：

- 公共 API 使用 `get_action(data) -> tuple[int, list[int]]` 和 `round_end(data) -> dict[str, object]`。
- 动作常量集中在 `jiujiang_ai.rules`。
- 牌常量集中在 `jiujiang_ai.tiles`。
