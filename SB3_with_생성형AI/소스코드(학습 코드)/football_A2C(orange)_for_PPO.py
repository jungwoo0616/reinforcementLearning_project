# -*- coding: utf-8 -*-
# file: orange_finetune_vs_blue3h.py
# 목적: ORANGE(A2C) 3시간 추가학습 - BLUE(PPO) 최신 전술에 대응하도록 미세조정
# - 오렌지 최신 모델(duala2c_latest.zip)에서 이어서 학습
# - 상대(블루 0/1)는 dualppo_latest.zip 스냅샷을 강하게 사용
# - 90분 x 2 페이즈(learn_id=2 → 3), 총 180분
# - 종료 시 models_dual_final_orange/duala2c_latest.zip 으로 원자적 저장

import os
import time
import glob
import datetime
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ===== 경로 설정(필요시 수정) =====
BLUE_LATEST_PATH   = os.path.join("PPO/models_dual_final_blue",   "dualppo_latest.zip")      # 블루 최신(PPO)
ORANGE_LATEST_PATH = os.path.join("A2C/models_dual_final_orange", "duala2c_latest.zip")      # 오렌지 최신(A2C)

SNAP_DIR   = "snapshots_dual_orange_finetune"
BEST_DIR   = "best_models_dual_orange_finetune"
EVAL_LOG   = "eval_logs_dual_orange_finetune"
FINAL_DIR  = "models_dual_final_orange"
FINAL_LATEST = os.path.join(FINAL_DIR, "duala2c_latest")  # .zip 자동 부착

os.makedirs(SNAP_DIR, exist_ok=True)
os.makedirs(os.path.join(BEST_DIR, "o2"), exist_ok=True)
os.makedirs(os.path.join(BEST_DIR, "o3"), exist_ok=True)
os.makedirs(os.path.join(EVAL_LOG, "o2"), exist_ok=True)
os.makedirs(os.path.join(EVAL_LOG, "o3"), exist_ok=True)
os.makedirs(FINAL_DIR, exist_ok=True)

# ===== 메모리/스레드 안정화 =====
os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# NumPy 1.24+ 호환
if not hasattr(np, "bool"):
    np.bool = np.bool_

import gym
from gym import spaces
import soccer_twos

# ------------------ 약한 정책 ------------------
class WeakPolicy:
    def __init__(self, action_space: spaces.Space, p_idle: float = 0.98, weak_action_index=None, seed: int = 0):
        self.action_space = action_space
        self.p_idle = float(p_idle)
        self.rng = np.random.default_rng(seed)
        if isinstance(action_space, spaces.Discrete):
            self.weak = int(0 if weak_action_index is None else weak_action_index)
        elif isinstance(action_space, spaces.MultiDiscrete):
            self.weak = np.zeros_like(action_space.nvec, dtype=np.int64) if weak_action_index is None \
                        else np.asarray(weak_action_index, dtype=np.int64)
        else:
            raise TypeError(f"Unsupported action_space: {type(action_space)}")

    def act(self):
        if self.rng.random() < self.p_idle:
            return self.weak.copy() if isinstance(self.weak, np.ndarray) else int(self.weak)
        return self.action_space.sample()

# ------------------ 스냅샷 정책(PPO→A2C 순서 로드) ------------------
class SnapshotPolicy:
    def __init__(self, path_zip: str, action_space: spaces.Space):
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        self.action_space = action_space
        self.model = None
        self.exp_dim = None
        try:
            from stable_baselines3 import PPO
            self.model = PPO.load(path_zip, env=None, device="cpu")
        except Exception:
            from stable_baselines3 import A2C
            self.model = A2C.load(path_zip, env=None, device="cpu")
        try:
            if hasattr(self.model, "observation_space") and self.model.observation_space is not None:
                self.exp_dim = int(np.prod(self.model.observation_space.shape))
            elif hasattr(self.model, "policy") and hasattr(self.model.policy, "observation_space"):
                self.exp_dim = int(np.prod(self.model.policy.observation_space.shape))
        except Exception:
            self.exp_dim = None

    @staticmethod
    def _with_onehot(obs_vec: np.ndarray, which: int = 0):
        one = np.array([1.0, 0.0], dtype=np.float32) if which == 0 else np.array([0.0, 1.0], dtype=np.float32)
        return np.concatenate([obs_vec.astype(np.float32, copy=False), one], axis=0)

    def _fix_obs_dim(self, obs_vec: np.ndarray) -> np.ndarray:
        x = np.asarray(obs_vec, dtype=np.float32)
        if self.exp_dim is None:
            if x.shape[0] == 336:
                return self._with_onehot(x, which=0)
            return x
        if x.shape[0] == self.exp_dim:
            return x
        if x.shape[0] + 2 == self.exp_dim:
            return self._with_onehot(x, which=0)
        if x.shape[0] - 2 == self.exp_dim:
            return x[:self.exp_dim]
        if x.shape[0] < self.exp_dim:
            pad = np.zeros(self.exp_dim - x.shape[0], dtype=np.float32)
            return np.concatenate([x, pad], axis=0)
        return x[:self.exp_dim]

    def act(self, obs_vec: np.ndarray, deterministic: bool = True):
        x = self._fix_obs_dim(obs_vec)
        a, _ = self.model.predict(x, deterministic=deterministic)
        if isinstance(self.action_space, spaces.Discrete):
            return int(a)
        return np.asarray(a, dtype=np.int64)

# ------------------ 정책 셀렉터 ------------------
class PolicyMixer:
    def __init__(self, action_space: spaces.Space, weak_p_idle: float,
                 latest_path: str, pool_paths: list, probs: dict, seed: int = 0):
        self.action_space = action_space
        self.probs = (probs or {}).copy()
        self.rng = np.random.default_rng(seed)

        self.weak = WeakPolicy(action_space, p_idle=weak_p_idle, seed=seed+777)

        self.latest = None
        if latest_path and os.path.isfile(latest_path):
            try:
                self.latest = SnapshotPolicy(latest_path, action_space)
            except Exception as e:
                print("[WARN] load latest snapshot failed:", e)

        self.pool = []
        for p in (pool_paths or []):
            if os.path.isfile(p):
                try:
                    self.pool.append(SnapshotPolicy(p, action_space))
                except Exception as e:
                    print("[WARN] load pool snapshot failed:", e)

        for k in ["latest", "pool", "weak"]:
            if k not in self.probs:
                self.probs[k] = 0.0
        s = sum(self.probs.values())
        self.probs = {k: (v / s if s > 0 else 0.0) for k, v in self.probs.items()}

    def _pick(self):
        r = self.rng.random()
        p_latest = self.probs.get("latest", 0.0)
        p_pool   = self.probs.get("pool", 0.0)
        if r < p_latest and self.latest is not None:
            return ("latest", self.latest)
        if r < p_latest + p_pool and len(self.pool) > 0:
            return ("pool", self.rng.choice(self.pool))
        return ("weak", self.weak)

    def act(self, obs_vec: np.ndarray):
        kind, pol = self._pick()
        if kind == "weak":
            return pol.act()
        return pol.act(obs_vec, deterministic=True)

# ------------------ Self-Play 학습 환경 (ORANGE 2/3 학습) ------------------
class SelfPlaySoccerEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self,
                 learn_agent_id: int,             # 2 또는 3
                 orange_latest: str,               # 오렌지 팀메이트
                 blue_latest: str,                 # 블루 상대
                 weak_p_idle: float = 0.97,
                 opponent_probs: dict = None,      # 블루(0/1)
                 teammate_probs: dict = None,      # 오렌지(2/3)
                 render: bool = False,
                 time_scale: float = 20.0,
                 file_name: str = None,
                 seed: int = 42,
                 worker_id: int = 0,
                 base_port: int = 9500):
        super().__init__()
        assert learn_agent_id in (2, 3)
        self.learn_id = int(learn_agent_id)

        self._env = soccer_twos.make(
            render=render, time_scale=time_scale, file_name=file_name,
            worker_id=worker_id, base_port=base_port
        )

        self.orig_obs_space: spaces.Box = self._env.observation_space
        self.action_space: spaces.Space = self._env.action_space

        low = np.full(self.orig_obs_space.shape[0] + 2, -np.inf, dtype=np.float32)
        high = np.full(self.orig_obs_space.shape[0] + 2, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # 기본 확률: 상대는 BLUE 최신을 강하게 사용
        opponent_probs = opponent_probs or {"latest": 0.90, "pool": 0.00, "weak": 0.10}
        teammate_probs = teammate_probs or {"latest": 0.70, "pool": 0.00, "weak": 0.30}

        self._base_weak_p = float(weak_p_idle)

        # 오렌지 팀메이트(2↔3)
        self._teammate = PolicyMixer(self.action_space, weak_p_idle=weak_p_idle,
                                     latest_path=orange_latest, pool_paths=[],
                                     probs=teammate_probs, seed=seed+1)
        # 블루 상대(0/1)
        self._opponent0 = PolicyMixer(self.action_space, weak_p_idle=weak_p_idle,
                                      latest_path=blue_latest, pool_paths=[],
                                      probs=opponent_probs, seed=seed+2)
        self._opponent1 = PolicyMixer(self.action_space, weak_p_idle=weak_p_idle,
                                      latest_path=blue_latest, pool_paths=[],
                                      probs=opponent_probs, seed=seed+3)

        self._obs_dict = None
        self.rng = np.random.default_rng(seed)

    def _aug_obs(self, obs_vec: np.ndarray) -> np.ndarray:
        # ORANGE: 2 -> [1,0], 3 -> [0,1]
        onehot = np.array([1.0, 0.0], dtype=np.float32) if self.learn_id == 2 else np.array([0.0, 1.0], dtype=np.float32)
        return np.concatenate([obs_vec.astype(np.float32, copy=False), onehot], axis=0)

    def _resample_difficulty(self):
        sampled = float(np.clip(np.random.normal(self._base_weak_p, 0.01), 0.88, 0.995))
        self._teammate.weak.p_idle = sampled
        self._opponent0.weak.p_idle = sampled
        self._opponent1.weak.p_idle = sampled

    def reset(self):
        self._resample_difficulty()
        self._obs_dict = self._env.reset()
        o = np.asarray(self._obs_dict[self.learn_id], dtype=np.float32)
        return self._aug_obs(o)

    def step(self, action):
        if isinstance(self.action_space, spaces.Discrete):
            aL = int(action)
        elif isinstance(self.action_space, spaces.MultiDiscrete):
            aL = np.asarray(action, dtype=np.int64)
        else:
            raise TypeError(f"Unsupported action_space: {type(self.action_space)}")

        if self.learn_id == 2:
            a2 = aL
            a3 = self._teammate.act(np.asarray(self._obs_dict[3], dtype=np.float32))
        else:
            a2 = self._teammate.act(np.asarray(self._obs_dict[2], dtype=np.float32))
            a3 = aL

        a0 = self._opponent0.act(np.asarray(self._obs_dict[0], dtype=np.float32))
        a1 = self._opponent1.act(np.asarray(self._obs_dict[1], dtype=np.float32))

        next_obs_d, rew_d, done_d, info_d = self._env.step({0: a0, 1: a1, 2: a2, 3: a3})
        self._obs_dict = next_obs_d

        obs_sel = np.asarray(next_obs_d[self.learn_id], dtype=np.float32)
        rew_sel = float(rew_d[self.learn_id])
        done = bool(done_d.get("__all__", False))
        info_sel = info_d.get(self.learn_id, {})

        if done:
            self._resample_difficulty()

        return self._aug_obs(obs_sel), rew_sel, done, info_sel

    def close(self):
        self._env.close()

# ------------------ 유틸 ------------------
def atomic_save_model(model, target_path_without_ext: str):
    os.makedirs(os.path.dirname(target_path_without_ext), exist_ok=True)
    tmp_base = target_path_without_ext + "_tmp"
    final_zip = target_path_without_ext + ".zip"
    tmp_zip = tmp_base + ".zip"
    try:
        model.save(tmp_base)          # -> tmp_base.zip 생성
        if os.path.exists(final_zip):
            os.replace(tmp_zip, final_zip)
        else:
            os.rename(tmp_zip, final_zip)
    finally:
        if os.path.exists(tmp_zip):
            try: os.remove(tmp_zip)
            except Exception: pass

# ------------------ 메인 ------------------
if __name__ == "__main__":
    from stable_baselines3 import A2C
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import EvalCallback
    import torch
    import gc

    # ===== 설정 =====
    try:
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        DEVICE = "cpu"
    print("[INFO] device:", DEVICE)

    try:
        torch.set_num_threads(1)
    except Exception:
        pass

    FILE_NAME = None
    SEED = 888
    BASE_PORT = 9500         # 블루 쪽과 포트 분리
    START_WORKER = 1600
    N_ENVS = 6
    TIME_SCALE = 20.0

    # 3시간(90분×2) 커리큘럼: learn_id=2 → 3
    CURRICULUM = [
        {"minutes": 90, "learn_id": 2, "weak_p_idle": 0.97},
        {"minutes": 90, "learn_id": 3, "weak_p_idle": 0.97},
    ]

    TEAMMATE_BASE_PROBS = {"latest": 0.70, "pool": 0.00, "weak": 0.30}
    OPP_BASE_PROBS      = {"latest": 0.90, "pool": 0.00, "weak": 0.10}  # BLUE 최신 강하게

    # ===== VecEnv 빌더 =====
    def make_env_fn(learn_id, orange_latest, blue_latest, weak_p_idle,
                    opp_probs, mate_probs, render, time_scale, file_name, seed, worker_id, base_port):
        def _init():
            return SelfPlaySoccerEnv(
                learn_agent_id=learn_id,
                orange_latest=orange_latest,
                blue_latest=blue_latest,
                weak_p_idle=weak_p_idle,
                opponent_probs=opp_probs,
                teammate_probs=mate_probs,
                render=render,
                time_scale=time_scale,
                file_name=file_name,
                seed=seed,
                worker_id=worker_id,
                base_port=base_port
            )
        return _init

    def build_vec_env(n_envs, start_worker_id, base_port,
                      learn_id, orange_latest, blue_latest, weak_p_idle, opp_probs, mate_probs,
                      file_name, seed, time_scale=20.0):
        env_fns = []
        for i in range(n_envs):
            wid = start_worker_id + i
            env_fns.append(
                make_env_fn(learn_id, orange_latest, blue_latest, weak_p_idle,
                            opp_probs, mate_probs, False, time_scale, file_name, seed + i, wid, base_port)
            )
        return DummyVecEnv(env_fns)

    def build_eval_env(start_worker_id, base_port,
                       learn_id, orange_latest, blue_latest, weak_p_idle, opp_probs, mate_probs,
                       file_name, seed, time_scale=20.0):
        wid = start_worker_id
        return DummyVecEnv([
            make_env_fn(learn_id, orange_latest, blue_latest, weak_p_idle,
                        opp_probs, mate_probs, False, time_scale, file_name, seed, wid, base_port)
        ])

    # ===== A2C 하이퍼 =====
    def lr_schedule(progress_remaining: float):
        # 3e-4 -> 1e-4 선형 감소
        return 1e-4 + (3e-4 - 1e-4) * progress_remaining

    A2C_KW = dict(
        n_steps=max(8, (5 * 256) // N_ENVS),  # 총 ~256 스텝 롤아웃
        learning_rate=lr_schedule,
        gamma=0.995,
        gae_lambda=0.95,
        ent_coef=0.008,            # 블루 전술 대응 위해 약간 낮춰 exploitation↑
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_rms_prop=True,
        rms_prop_eps=1e-5,
        normalize_advantage=True,
        policy_kwargs=dict(net_arch=[512, 256]),
        verbose=1,
        seed=SEED,
        device=DEVICE,
        tensorboard_log="runs/a2c_finetune_orange_vs_blue3h",
    )

    # ===== 초기 환경 =====
    init = CURRICULUM[0]
    env = build_vec_env(
        n_envs=N_ENVS, start_worker_id=START_WORKER, base_port=BASE_PORT,
        learn_id=init["learn_id"], orange_latest=ORANGE_LATEST_PATH, blue_latest=BLUE_LATEST_PATH,
        weak_p_idle=init["weak_p_idle"], opp_probs=OPP_BASE_PROBS, mate_probs=TEAMMATE_BASE_PROBS,
        file_name=FILE_NAME, seed=SEED, time_scale=TIME_SCALE
    )

    # ===== 오렌지 최신 모델 로드(필수) =====
    assert os.path.isfile(ORANGE_LATEST_PATH), f"ORANGE_LATEST_PATH not found: {ORANGE_LATEST_PATH}"
    from stable_baselines3 import A2C
    model = A2C.load(ORANGE_LATEST_PATH, env=env, device=DEVICE)
    model.set_env(env)

    # (선택) 간단한 평가 env와 콜백
    eval_env_o2 = build_eval_env(
        start_worker_id=START_WORKER + 999, base_port=BASE_PORT,
        learn_id=2, orange_latest=ORANGE_LATEST_PATH, blue_latest=BLUE_LATEST_PATH,
        weak_p_idle=0.95, opp_probs=OPP_BASE_PROBS, mate_probs=TEAMMATE_BASE_PROBS,
        file_name=FILE_NAME, seed=SEED+111, time_scale=TIME_SCALE
    )
    eval_env_o3 = build_eval_env(
        start_worker_id=START_WORKER + 1000, base_port=BASE_PORT,
        learn_id=3, orange_latest=ORANGE_LATEST_PATH, blue_latest=BLUE_LATEST_PATH,
        weak_p_idle=0.95, opp_probs=OPP_BASE_PROBS, mate_probs=TEAMMATE_BASE_PROBS,
        file_name=FILE_NAME, seed=SEED+222, time_scale=TIME_SCALE
    )

    from stable_baselines3.common.callbacks import EvalCallback
    eval_cb_o2 = EvalCallback(
        eval_env=eval_env_o2, best_model_save_path=os.path.join(BEST_DIR, "o2"),
        log_path=os.path.join(EVAL_LOG, "o2"),
        eval_freq=100_000, n_eval_episodes=6, deterministic=True, render=False
    )
    eval_cb_o3 = EvalCallback(
        eval_env=eval_env_o3, best_model_save_path=os.path.join(BEST_DIR, "o3"),
        log_path=os.path.join(EVAL_LOG, "o3"),
        eval_freq=100_000, n_eval_episodes=6, deterministic=True, render=False
    )

    # ===== 3시간 학습 루프 =====
    for phase_idx, ph in enumerate(CURRICULUM, start=1):
        learn_id = ph["learn_id"]
        weak_p   = ph["weak_p_idle"]
        minutes  = ph["minutes"]

        # 환경 재빌드(learn_id 전환)
        try:
            env.close()
        except Exception:
            pass
        del env
        gc.collect()
        time.sleep(0.5)

        env = build_vec_env(
            n_envs=N_ENVS, start_worker_id=START_WORKER + phase_idx * 32, base_port=BASE_PORT,
            learn_id=learn_id, orange_latest=ORANGE_LATEST_PATH, blue_latest=BLUE_LATEST_PATH,
            weak_p_idle=weak_p, opp_probs=OPP_BASE_PROBS, mate_probs=TEAMMATE_BASE_PROBS,
            file_name=FILE_NAME, seed=SEED + phase_idx * 101, time_scale=TIME_SCALE
        )
        model.set_env(env)

        print(f"\n[PHASE] {phase_idx}/{len(CURRICULUM)}  learn_id={learn_id}  weak_p={weak_p:.3f}  minutes={minutes}  vs=BLUE(PPO latest, p≈{OPP_BASE_PROBS['latest']})")
        t0 = time.time()
        while time.time() - t0 < minutes * 60:
            model.learn(total_timesteps=120_000, reset_num_timesteps=False, callback=[eval_cb_o2, eval_cb_o3])
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{now}] phase {phase_idx} elapsed={(time.time()-t0)/60:.1f} min")

        # 중간 스냅샷(선택)
        os.makedirs(SNAP_DIR, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        snap_path = os.path.join(SNAP_DIR, f"duala2c_finetune_{stamp}")
        model.save(snap_path)
        print("[SNAPSHOT SAVED]", snap_path + ".zip")

    # ----- 종료 정리 -----
    try:
        env.close()
    except Exception:
        pass
    try:
        eval_env_o2.close(); eval_env_o3.close()
    except Exception:
        pass

    # 최신 모델로 원자적 저장
    try:
        atomic_save_model(model, FINAL_LATEST)
        print(f"[SAVE-LAST] {FINAL_LATEST}.zip")
    except Exception as e:
        print(f"[WARN] final latest save failed: {repr(e)}")

    print("[DONE] Orange A2C 3-hour finetune vs Blue PPO latest finished.")
