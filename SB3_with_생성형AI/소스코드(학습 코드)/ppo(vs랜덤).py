# -*- coding: utf-8 -*-
import ray
from ray import tune
from soccer_twos import EnvType

from utils import create_rllib_env

NUM_ENVS_PER_WORKER = 5

if __name__ == "__main__":
    # 윈도우 로컬 환경: 대시보드 비활성화
    ray.init(include_dashboard=False)

    # 환경 등록
    tune.registry.register_env("Soccer", create_rllib_env)

    analysis = tune.run(
        "PPO",
        name="PPO_1",
        config={
            # system settings
            "num_gpus": 1,
            "num_workers": 0,                    # 로컬 워커 1개
            "num_envs_per_worker": NUM_ENVS_PER_WORKER,
            "log_level": "INFO",
            "framework": "torch",

            # RL setup
            "env": "Soccer",
            "env_config": {
                "num_envs_per_worker": NUM_ENVS_PER_WORKER,
                "variation": EnvType.team_vs_policy,
                "multiagent": False,
            },

            "model": {
                "vf_share_layers": True,
                "fcnet_hiddens": [512, 512],
            },
        },
        stop={
            "timesteps_total": 20000000,
            "time_total_s": 14400,  # 필요하면 시간 제한도 추가 가능
        },
        checkpoint_freq=10,
        checkpoint_at_end=True,
        local_dir="./ray_results",
        # restore="./ray_results/PPO_selfplay_1/PPO_Soccer_ID/checkpoint_00X/checkpoint-X",
    )

    best_trial = analysis.get_best_trial("episode_reward_mean", mode="max")
    print(best_trial)

    best_checkpoint = analysis.get_best_checkpoint(
        trial=best_trial, metric="episode_reward_mean", mode="max"
    )
    print(best_checkpoint)
    print("Done training")
