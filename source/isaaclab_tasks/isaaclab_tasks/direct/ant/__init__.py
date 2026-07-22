# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Ant locomotion environment.
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Ant-Direct-v0",
    entry_point=f"{__name__}.ant_env:AntEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ant_env:AntEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Ant-Direct-vTransformer",
    entry_point=f"{__name__}.ant_env:AntEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ant_env:AntEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_transformer_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Ant-Direct-vLSTM",
    entry_point=f"{__name__}.ant_env:AntEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ant_env:AntEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_lstm_ppo_cfg.yaml",
        "skrl_ppo_rnn_cfg_entry_point": f"{agents.__name__}:skrl_lstm_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Ant-Direct-vMLPChunk",
    entry_point=f"{__name__}.ant_env:AntEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ant_env:AntEnvCfg",
        "skrl_ppo_chunked_cfg_entry_point": f"{agents.__name__}:skrl_mlp_chunked_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Ant-Direct-vMax",
    entry_point=f"{__name__}.ant_env:AntEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ant_env:AntEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_max_reward_cfg.yaml",
    },
)
