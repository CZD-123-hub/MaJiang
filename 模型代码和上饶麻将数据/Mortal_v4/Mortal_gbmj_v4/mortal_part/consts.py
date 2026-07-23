"""
Constants for the v4 supervised policy project.

The action space keeps the legacy 235-way convention for the policy head,
legal-action masks, and arena compatibility.

The default observation is the v4 visible encoder without the 11-channel
paper-style foresight block:

- longer 28-turn discard history
- visible/remaining tile counts
- shanten, waits, discard-progress hints already maintained by PlayerState
- current action context
- compact table scalar planes

[V4 direct-235 no-foresight] Expectation-value/SPCalculator and the later
11-channel foresight planes are both disabled for the supervised baseline.
Old 205-channel checkpoints are still handled by obs version compatibility.
"""

# 前 194 个平面沿用原监督模型；第 195 个平面标出上饶精牌位置。
obs_shape = (195, 4, 9)
oracle_obs_shape = (92, 4, 9)

ACTION_SPACE = 235

PASS_INDEX = 0
DISCARD_BASE = 1
DISCARD_COUNT = 34
WIN_INDEX = 35
MINGGANG_BASE = 36
MINGGANG_COUNT = 34
ANKANG_BASE = 70
ANKANG_COUNT = 34
ADDKONG_BASE = 104
ADDKONG_COUNT = 34
PUNG_BASE = 138
PUNG_COUNT = 34
CHOW_BASE = 172
CHOW_COUNT = 63

# The paper's printed action ranges contain off-by-one inconsistencies for
# AddKong and Chow. We follow the stated class counts and total dimension 235.

GRP_SIZE = 6
MAX_TSUMOS_LEFT = 21
