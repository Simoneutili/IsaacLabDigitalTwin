# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import pathlib
import sys

# Allow for import of items from the ray workflow.
CUR_DIR = pathlib.Path(__file__).parent
UTIL_DIR = CUR_DIR.parent
sys.path.extend([str(UTIL_DIR), str(CUR_DIR)])
import util
import param_sweep
from ray import tune


class TiAGO_SB3SAC_JobCfg(param_sweep.SB3SAC_JobCfg):
    def __init__(self, cfg: dict = {}):
        cfg = util.populate_isaac_ray_cfg_args(cfg)
        cfg["runner_args"]["--task"] = tune.choice(["Isaac-DigitalTwin-TiAGO-v0"])
        super().__init__(cfg, vary_env_count=True, vary_activation_fn=False, vary_net_arch=False)
        
class TiAGO_SB3PPO_JobCfg(param_sweep.SB3PPO_JobCfg):
    def __init__(self, cfg: dict = {}):
        cfg = util.populate_isaac_ray_cfg_args(cfg)
        cfg["runner_args"]["--task"] = tune.choice(["Isaac-DigitalTwin-TiAGO-v0"])
        super().__init__(cfg, vary_env_count=True, vary_activation_fn=False, vary_net_arch=False)
        
        
class TiAGO_RLGAMESPPO_JobCfg(param_sweep.RLGAMESPPO_JobCfg):
    def __init__(self, cfg: dict = {}):
        cfg = util.populate_isaac_ray_cfg_args(cfg)
        cfg["runner_args"]["--task"] = tune.choice(["Isaac-DigitalTwin-TiAGO-v0"])
        super().__init__(cfg, vary_env_count=True, vary_cnn=False, vary_mlp=True)
        
