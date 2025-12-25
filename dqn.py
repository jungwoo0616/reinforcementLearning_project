import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.warn = lambda *args, **kwargs: None

import yaml
import numpy as np

import ray
from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.tune.registry import register_env

from utils import create_rllib_env, sample_pos_vel, sample_player

NUM_ENVS_PER_WORKER = 3

CURRICULUM_REW_THRESH = -0.23
SELFPLAY_REW_THRESH   = -0.25

current = 0

with open("curriculum.yaml", "r", encoding="utf-8") as f:
    curriculum = yaml.load(f, Loader=yaml.FullLoader)
tasks = curriculum["tasks"]

config_fns = {
    "none": lambda *_: None,
    "random_players": lambda env: None,
}

def policy_mapping_fn(agent_id, *args, **kwargs):
    if agent_id == 0:
        return "default"
    else:
        return np.random.choice(
            ["default", "opponent_1", "opponent_2", "opponent_3"],
            size=1,
            p=[0.50, 0.25, 0.125, 0.125],
        )[0]

class CurriculumSelfPlayUpdateCallback(DefaultCallbacks):
    def on_episode_start(
        self, *, worker, base_env, policies, episode, env_index, **kwargs
    ) -> None:
        global current, tasks
        for env in base_env.get_unwrapped():
            cfg_name = tasks[current]["config_fn"]
            config_fns[cfg_name](env)
            env.env_channel.set_parameters(
                ball_state=sample_pos_vel(tasks[current]["ranges"]["ball"]),
                players_states={
                    player: sample_player(tasks[current]["ranges"]["players"][player])
                    for player in tasks[current]["ranges"]["players"]
                },
            )

    def on_train_result(self, **info):
        global current, tasks

        trainer = info["trainer"]
        mean_rew = info["result"]["episode_reward_mean"]

        if mean_rew > CURRICULUM_REW_THRESH:
            if current < len(tasks) - 1:
                print("---- Updating curriculum task!!! ----")
                current += 1
                print(f"Current task: {current} - {tasks[current]['name']}")

        if mean_rew > SELFPLAY_REW_THRESH:
            print("---- Updating opponents (self-play)!!! ----")
            trainer.set_weights(
                {
                    "opponent_3": trainer.get_weights(["opponent_2"])["opponent_2"],
                    "opponent_2": trainer.get_weights(["opponent_1"])["opponent_1"],
                    "opponent_1": trainer.get_weights(["default"])["default"],
                }
            )

if __name__ == "__main__":
    ray.init(include_dashboard=False)

    register_env("Soccer", create_rllib_env)

    # DQN이 MultiDiscrete를 사용하지 못함 -> flatten_branched=True로 설정
    temp_env = create_rllib_env({
        "num_envs_per_worker": NUM_ENVS_PER_WORKER,
        "flatten_branched": True,
    })
    obs_space = temp_env.observation_space
    act_space = temp_env.action_space
    temp_env.close()

    analysis = tune.run(
        "DQN",
        name="DQN_selfplay_twos",
        config={
            "num_gpus": 1,
            "num_workers": 0,
            "num_envs_per_worker": NUM_ENVS_PER_WORKER,
            "log_level": "INFO",
            "framework": "torch",
            "callbacks": CurriculumSelfPlayUpdateCallback,

            "multiagent": {
                "policies": {
                    "default":    (None, obs_space, act_space, {}),
                    "opponent_1": (None, obs_space, act_space, {}),
                    "opponent_2": (None, obs_space, act_space, {}),
                    "opponent_3": (None, obs_space, act_space, {}),
                },
                "policy_mapping_fn": tune.function(policy_mapping_fn),
                "policies_to_train": ["default"],
            },

            "env": "Soccer",
            "env_config": {
                "num_envs_per_worker": NUM_ENVS_PER_WORKER,
                "flatten_branched": True,
            },

            "model": {
                "fcnet_hiddens": [256, 256],
                "fcnet_activation": "relu",
            },

            "lr": 1e-4,
            "train_batch_size": 64,
            "learning_starts": 10000,
            "buffer_size": 500000,
            "target_network_update_freq": 5000,
            "gamma": 0.99,
            "n_step": 1,
            "double_q": True,
            "dueling": True,

            "rollout_fragment_length": 5000,
        },
        stop={
            "timesteps_total": 10000000,
        },
        checkpoint_freq=10,
        checkpoint_at_end=True,
        local_dir="./ray_results(DQN)",
    )

    best_trial = analysis.get_best_trial("episode_reward_mean", mode="max")
    print(best_trial)

    best_checkpoint = analysis.get_best_checkpoint(
        trial=best_trial, metric="episode_reward_mean", mode="max"
    )
    print(best_checkpoint)
    print("Done training")
