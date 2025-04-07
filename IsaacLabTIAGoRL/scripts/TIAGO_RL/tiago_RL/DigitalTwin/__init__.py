# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
TiAGO digital twin environment environment.
"""

import gymnasium as gym

from . import agents
from .tiago_env import TiagoEnv, TiagoEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="Isaac-DigitalTwin-TiAGO-v0",
    entry_point="tiago_RL.DigitalTwin:TiagoEnv",  # Path to the class, ensure it's correct
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": TiagoEnvCfg,
        "sb3_ppo_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
        "sb3_sac_cfg_entry_point": f"{agents.__name__}:sb3_sac_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

print ("[INFO]: Registered Isaac-DigitalTwin-TiAGO-v0 environment.")

