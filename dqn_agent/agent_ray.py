# -*- coding: utf-8 -*-
import pickle
import os
from typing import Dict

import gym
import numpy as np
import ray
from ray import tune
from ray.rllib.env.base_env import BaseEnv
from ray.tune.registry import get_trainable_cls

from soccer_twos import AgentInterface

ALGORITHM = "DQN"

CHECKPOINT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    r"./ray_results(DQN)/DQN_selfplay_twos\DQN_Soccer_2ef1a_00000_0_2025-11-23_21-51-52\checkpoint_000050/checkpoint-50",
)

POLICY_NAME = "default"


class RayAgent(AgentInterface):
    def __init__(self, env: gym.Env):
        super().__init__()
        ray.init(ignore_reinit_error=True)
        self.name = "DQN-Agent"

        config_path = ""
        if CHECKPOINT_PATH:
            config_dir = os.path.dirname(CHECKPOINT_PATH)
            config_path = os.path.join(config_dir, "params.pkl")
            if not os.path.exists(config_path):
                config_path = os.path.join(config_dir, "../params.pkl")

        if os.path.exists(config_path):
            with open(config_path, "rb") as f:
                config = pickle.load(f)
        else:
            raise ValueError("Could not find params.pkl in either the checkpoint dir or its parent directory!")

        config["num_workers"] = 0
        config["num_gpus"] = 0
        config["explore"] = False

        tune.registry.register_env("DummyEnv", lambda *_: BaseEnv())
        config["env"] = "DummyEnv"

        cls = get_trainable_cls(ALGORITHM)
        agent = cls(env=config["env"], config=config)

        if not os.path.exists(CHECKPOINT_PATH):
            raise FileNotFoundError(f"Checkpoint file not found: {CHECKPOINT_PATH}")

        with open(CHECKPOINT_PATH, "rb") as f:
            checkpoint_data = pickle.load(f)

        if "worker" not in checkpoint_data:
            raise KeyError(f"'worker' key not found in checkpoint data. Available keys: {list(checkpoint_data.keys())}")

        worker_blob = checkpoint_data["worker"]

        if isinstance(worker_blob, (bytes, bytearray)):
            worker_state = pickle.loads(worker_blob)
        else:
            worker_state = worker_blob

        if not isinstance(worker_state, dict):
            raise TypeError(f"Unexpected type for worker_state: {type(worker_state)}.")

        if "state" in worker_state and isinstance(worker_state["state"], dict):
            policy_container = worker_state["state"]
        elif "policy_states" in worker_state and isinstance(worker_state["policy_states"], dict):
            policy_container = worker_state["policy_states"]
        else:
            raise KeyError(f"Could not find policy states in checkpoint worker data. Keys: {list(worker_state.keys())}")

        if POLICY_NAME not in policy_container:
            raise KeyError(f"Policy '{POLICY_NAME}' not found in checkpoint. Available: {list(policy_container.keys())}")

        policy_state = policy_container[POLICY_NAME]

        if isinstance(policy_state, dict) and "weights" in policy_state:
            weights_dict = policy_state["weights"]
        elif isinstance(policy_state, dict):
            weights_dict = {
                k: v for k, v in policy_state.items()
                if not k.startswith("_optimizer") and not k.startswith("optimizer")
            }
        else:
            raise TypeError(f"Unexpected type for policy_state: {type(policy_state)}")

        policy = agent.get_policy(POLICY_NAME)
        policy.set_weights(weights_dict)

        self.policy = policy

    @staticmethod
    def _unflatten_action(flat_action: int) -> np.ndarray:
        a = int(flat_action)
        a0 = a // 9
        a1 = (a // 3) % 3
        a2 = a % 3
        return np.array([a0, a1, a2], dtype=np.int64)

    def act(self, observation: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        actions = {}
        for player_id in observation:
            flat_act, *_ = self.policy.compute_single_action(observation[player_id])

            if np.isscalar(flat_act):
                actions[player_id] = self._unflatten_action(flat_act)
            else:
                actions[player_id] = np.asarray(flat_act, dtype=np.int64)

        return actions
