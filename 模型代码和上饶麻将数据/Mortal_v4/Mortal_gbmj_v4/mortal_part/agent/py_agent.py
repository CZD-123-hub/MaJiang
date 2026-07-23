from typing import List

from mortal_part.agent.defs import BatchAgent
from mortal_part.agent.supervised import SupervisedBatchAgent
from rl.agent import PPOBatchAgent


def new_py_agent(engine, player_ids: List[int]) -> BatchAgent:
    # [V4 local-play] The copied arena calls this factory.  v4 uses a
    # supervised 235-way policy engine instead of Mortal's Brain+DQN.
    if getattr(engine, "engine_type", None) == "supervised":
        return SupervisedBatchAgent(engine, player_ids)
    if getattr(engine, "engine_type", None) == "ppo":
        # [V4 PPO RL] Allow arena factories to create rollout-recording PPO agents.
        return PPOBatchAgent(engine, player_ids)
    raise ValueError("unsupported engine type for v4 local play: %r" % (getattr(engine, "engine_type", None),))
