# -*- coding: utf-8 -*-
# file: soccer_train_dual_selfplay_6h_blue_safe_dqn.py
# 목적: Dual DQN Self-Play (BLUE 0/1) - 6시간 커리큘럼(안정성 강화 그대로)
# - 핵심 구조/로직/커리큘럼/스냅샷/포트 처리 전부 동일
# - PPO -> DQN 교체
# - DQN은 Discrete만 지원하므로 학습자 액션만 Discrete로 플래튼 옵션 추가(원 환경은 그대로 MultiDiscrete 허용)

import os
import time
import glob
import datetime
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ===== 메모리/스레드 안정화 환경변수 =====
os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)  # expandable_segments 유발 제거
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

# ------------------ 스냅샷 정책 (CPU 로드 + 관측차원 자동보정) ------------------
class SnapshotPolicy:
    def __init__(self, path_zip: str, action_space: spaces.Space):
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        from stable_baselines3 import PPO
        self.model = PPO.load(path_zip, env=None, device="cpu")
        self.action_space = action_space
        self.exp_dim = None
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
        p_pool = self.probs.get("pool", 0.0)
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

# ---------- MultiDiscrete <-> Discrete 변환 유틸 (학습자 전용) ----------
def _md_index_to_vec(index: int, nvec: np.ndarray) -> np.ndarray:
    idx = int(index)
    out = np.zeros_like(nvec, dtype=np.int64)
    for i in range(len(nvec)-1, -1, -1):
        out[i] = idx % nvec[i]
        idx //= nvec[i]
    return out

def _md_vec_to_index(vec: np.ndarray, nvec: np.ndarray) -> int:
    idx = 0
    for i, base in enumerate(nvec):
        idx = idx * int(base) + int(vec[i])
    return int(idx)

# ------------------ Dual Self-Play 학습 환경 ------------------
class SelfPlaySoccerEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self,
                 learn_agent_id: int,
                 latest_snapshot: str,
                 pool_snapshots: list,
                 weak_p_idle: float = 0.97,
                 opponent_probs: dict = None,
                 teammate_probs: dict = None,
                 render: bool = False,
                 time_scale: float = 20.0,
                 file_name: str = None,
                 seed: int = 42,
                 worker_id: int = 0,
                 base_port: int = 6005,
                 flatten_for_dqn: bool = False):   # ★ 추가: 학습자 액션 Discrete 플래튼
        super().__init__()
        assert learn_agent_id in (0, 1)
        self.learn_id = int(learn_agent_id)

        self._env = soccer_twos.make(
            render=render, time_scale=time_scale, file_name=file_name,
            worker_id=worker_id, base_port=base_port
        )

        self.orig_obs_space: spaces.Box = self._env.observation_space
        self.orig_action_space: spaces.Space = self._env.action_space

        # 관측 공간(+2 one-hot)
        low = np.full(self.orig_obs_space.shape[0] + 2, -np.inf, dtype=np.float32)
        high = np.full(self.orig_obs_space.shape[0] + 2, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # 학습자 액션 플래튼 설정
        self.flatten_for_dqn = bool(flatten_for_dqn and isinstance(self.orig_action_space, spaces.MultiDiscrete))
        if self.flatten_for_dqn:
            self._nvec = np.asarray(self.orig_action_space.nvec, dtype=np.int64)
            self.action_space = spaces.Discrete(int(np.prod(self._nvec)))
        else:
            self.action_space = self.orig_action_space

        opponent_probs = opponent_probs or {"latest": 0.0, "pool": 0.0, "weak": 1.0}
        teammate_probs = teammate_probs or {"latest": 0.60, "pool": 0.30, "weak": 0.10}

        self._base_weak_p = float(weak_p_idle)

        # ★ 팀메이트/상대는 항상 "원래 액션공간"으로 정책 생성 (MultiDiscrete 유지)
        self._teammate = PolicyMixer(self.orig_action_space, weak_p_idle=weak_p_idle,
                                     latest_path=latest_snapshot, pool_paths=pool_snapshots,
                                     probs=teammate_probs, seed=seed+1)
        self._opponent2 = PolicyMixer(self.orig_action_space, weak_p_idle=weak_p_idle,
                                      latest_path=latest_snapshot, pool_paths=pool_snapshots,
                                      probs=opponent_probs, seed=seed+2)
        self._opponent3 = PolicyMixer(self.orig_action_space, weak_p_idle=weak_p_idle,
                                      latest_path=latest_snapshot, pool_paths=pool_snapshots,
                                      probs=opponent_probs, seed=seed+3)

        self._obs_dict = None
        self.rng = np.random.default_rng(seed)

    def _aug_obs(self, obs_vec: np.ndarray) -> np.ndarray:
        onehot = np.array([1.0, 0.0], dtype=np.float32) if self.learn_id == 0 else np.array([0.0, 1.0], dtype=np.float32)
        return np.concatenate([obs_vec.astype(np.float32, copy=False), onehot], axis=0)

    def _resample_difficulty(self):
        sampled = float(np.clip(np.random.normal(self._base_weak_p, 0.01), 0.88, 0.995))
        self._teammate.weak.p_idle = sampled
        self._opponent2.weak.p_idle = sampled
        self._opponent3.weak.p_idle = sampled

    def reset(self):
        self._resample_difficulty()
        self._obs_dict = self._env.reset()
        o = np.asarray(self._obs_dict[self.learn_id], dtype=np.float32)
        return self._aug_obs(o)

    def step(self, action):
        # ----- 학습자 액션 처리 -----
        if self.flatten_for_dqn and isinstance(self.orig_action_space, spaces.MultiDiscrete):
            # DQN이 준 Discrete index -> MultiDiscrete 벡터로 복원
            aL = _md_index_to_vec(int(action), self._nvec)
        else:
            if isinstance(self.orig_action_space, spaces.Discrete):
                aL = int(action)
            elif isinstance(self.orig_action_space, spaces.MultiDiscrete):
                aL = np.asarray(action, dtype=np.int64)
            else:
                raise TypeError(f"Unsupported action_space: {type(self.orig_action_space)}")

        # ----- 팀메이트/상대 액션 (항상 원래 공간으로) -----
        if self.learn_id == 0:
            a0 = aL
            a1 = self._teammate.act(np.asarray(self._obs_dict[1], dtype=np.float32))
        else:
            a0 = self._teammate.act(np.asarray(self._obs_dict[0], dtype=np.float32))
            a1 = aL

        a2 = self._opponent2.act(np.asarray(self._obs_dict[2], dtype=np.float32))
        a3 = self._opponent3.act(np.asarray(self._obs_dict[3], dtype=np.float32))

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
def list_snapshots(dir_path, pattern="*.zip"):
    if not os.path.isdir(dir_path):
        return []
    return sorted(glob.glob(os.path.join(dir_path, pattern)))

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
            try:
                os.remove(tmp_zip)
            except Exception:
                pass

# ------------------ 메인 ------------------
if __name__ == "__main__":
    from stable_baselines3 import DQN
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
    SEED = 42
    BASE_PORT = 9300
    START_WORKER = 640
    N_ENVS = 6
    TIME_SCALE = 20.0

    # DQN 학습자에 대해서만 Discrete 플래튼
    FLATTEN_FOR_DQN = True  # ★ 핵심 스위치 (나머지는 동일)

    SNAP_DIR = "snapshots_dual_blue"
    BEST_DIR = "best_models_dual_blue"
    EVAL_LOG = "eval_logs_dual_blue"
    FINAL_DIR = "models_dual_final_blue"   # ★ 최신 모델 저장 디렉터리
    FINAL_LATEST = os.path.join(FINAL_DIR, "dualppo_latest")  # 파일명 호환 유지
    os.makedirs(SNAP_DIR, exist_ok=True)
    os.makedirs(os.path.join(BEST_DIR, "b0"), exist_ok=True)
    os.makedirs(os.path.join(BEST_DIR, "b1"), exist_ok=True)
    os.makedirs(os.path.join(EVAL_LOG, "b0"), exist_ok=True)
    os.makedirs(os.path.join(EVAL_LOG, "b1"), exist_ok=True)
    os.makedirs(FINAL_DIR, exist_ok=True)

    # ===== 6시간 커리큘럼 ===== (동일)
    CURRICULUM = [
        {"minutes": 8, "learn_id": 0, "weak_p_idle": 0.995},
        {"minutes": 8, "learn_id": 1, "weak_p_idle": 0.992},
        {"minutes": 8, "learn_id": 0, "weak_p_idle": 0.990},
        {"minutes": 8, "learn_id": 1, "weak_p_idle": 0.988},
        {"minutes": 8, "learn_id": 0, "weak_p_idle": 0.985},

        {"minutes": 20, "learn_id": 1, "weak_p_idle": 0.975},
        {"minutes": 20, "learn_id": 0, "weak_p_idle": 0.975},
        {"minutes": 20, "learn_id": 1, "weak_p_idle": 0.970},
        {"minutes": 20, "learn_id": 0, "weak_p_idle": 0.970},
        {"minutes": 20, "learn_id": 1, "weak_p_idle": 0.968},
        {"minutes": 20, "learn_id": 0, "weak_p_idle": 0.968},
        {"minutes": 20, "learn_id": 1, "weak_p_idle": 0.965},
        {"minutes": 20, "learn_id": 0, "weak_p_idle": 0.965},
        {"minutes": 20, "learn_id": 1, "weak_p_idle": 0.963},
        {"minutes": 20, "learn_id": 0, "weak_p_idle": 0.963},

        {"minutes": 10, "learn_id": 0, "weak_p_idle": 0.955},
        {"minutes": 10, "learn_id": 1, "weak_p_idle": 0.955},
        {"minutes": 10, "learn_id": 0, "weak_p_idle": 0.950},
        {"minutes": 10, "learn_id": 1, "weak_p_idle": 0.950},
        {"minutes": 10, "learn_id": 0, "weak_p_idle": 0.948},
        {"minutes": 10, "learn_id": 1, "weak_p_idle": 0.948},

        {"minutes": 15, "learn_id": 0, "weak_p_idle": 0.945},
        {"minutes": 15, "learn_id": 1, "weak_p_idle": 0.940},
        {"minutes": 15, "learn_id": 0, "weak_p_idle": 0.938},
        {"minutes": 15, "learn_id": 1, "weak_p_idle": 0.935},
    ]

    TEAMMATE_BASE_PROBS = {"latest": 0.60, "pool": 0.30, "weak": 0.10}

    def opponent_probs_for_phase(phase_idx: int):
        if phase_idx <= 5:
            return {"latest": 0.0,  "pool": 0.0,  "weak": 1.0}
        elif phase_idx <= 10:
            return {"latest": 0.10, "pool": 0.10, "weak": 0.80}
        elif phase_idx <= 15:
            return {"latest": 0.15, "pool": 0.20, "weak": 0.65}
        elif phase_idx <= 21:
            return {"latest": 0.20, "pool": 0.25, "weak": 0.55}
        else:
            return {"latest": 0.25, "pool": 0.30, "weak": 0.45}

    # ===== VecEnv 빌더 (DummyVecEnv) =====
    def make_env_fn(learn_id, latest_snapshot, pool_paths, weak_p_idle, opp_probs, mate_probs,
                    render, time_scale, file_name, seed, worker_id, base_port, flatten_for_dqn):
        def _init():
            return SelfPlaySoccerEnv(
                learn_agent_id=learn_id,
                latest_snapshot=latest_snapshot,
                pool_snapshots=pool_paths,
                weak_p_idle=weak_p_idle,
                opponent_probs=opp_probs,
                teammate_probs=mate_probs,
                render=render,
                time_scale=time_scale,
                file_name=file_name,
                seed=seed,
                worker_id=worker_id,
                base_port=base_port,
                flatten_for_dqn=flatten_for_dqn,  # ★
            )
        return _init

    def _build_vec_env_with_retry(n_envs, start_worker_id, base_port,
                                  learn_id, latest_snapshot, pool_paths, weak_p_idle, opp_probs, mate_probs,
                                  file_name, seed, time_scale=20.0, max_attempts=4, flatten_for_dqn=True):
        from stable_baselines3.common.vec_env import DummyVecEnv
        block_shift = 0
        attempt = 1
        while True:
            try:
                env_fns = []
                for i in range(n_envs):
                    wid = start_worker_id + block_shift + i
                    env_fns.append(
                        make_env_fn(learn_id, latest_snapshot, pool_paths, weak_p_idle, opp_probs, mate_probs,
                                    False, time_scale, file_name, seed + i, wid, base_port, flatten_for_dqn)
                    )
                return DummyVecEnv(env_fns)
            except Exception as e:
                print(f"[WARN] build_vec_env attempt={attempt} failed: {repr(e)}")
                if attempt >= max_attempts:
                    raise
                time.sleep(1.5)
                block_shift += (n_envs + 4)
                attempt += 1

    def build_vec_env(n_envs, start_worker_id, base_port,
                      learn_id, latest_snapshot, pool_paths, weak_p_idle, opp_probs, mate_probs,
                      file_name, seed, time_scale=20.0, flatten_for_dqn=True):
        return _build_vec_env_with_retry(
            n_envs, start_worker_id, base_port,
            learn_id, latest_snapshot, pool_paths, weak_p_idle, opp_probs, mate_probs,
            file_name, seed, time_scale, flatten_for_dqn=flatten_for_dqn
        )

    def build_eval_env(start_worker_id, base_port,
                       learn_id, latest_snapshot, pool_paths, weak_p_idle, opp_probs, mate_probs,
                       file_name, seed, time_scale=20.0, flatten_for_dqn=True):
        from stable_baselines3.common.vec_env import DummyVecEnv
        wid = start_worker_id
        return DummyVecEnv([
            make_env_fn(learn_id, latest_snapshot, pool_paths, weak_p_idle, opp_probs, mate_probs,
                        False, time_scale, file_name, seed, wid, base_port, flatten_for_dqn)
        ])

    # ===== DQN 하이퍼 =====
    DQN_KW = dict(
        learning_rate=1e-4,
        buffer_size=200_000,
        learning_starts=50_000,
        batch_size=1024,
        tau=0.01,
        gamma=0.995,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=10_000,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        exploration_fraction=0.2,
        policy_kwargs=dict(net_arch=[1024, 512]),
        verbose=1,
        seed=SEED,
        device=DEVICE,
        tensorboard_log="runs/dqn_dual_selfplay_6h_blue",
    )

    # ===== 초기 환경/모델 =====
    latest_snapshot = None
    pool_snapshots = []

    init = CURRICULUM[0]
    env = build_vec_env(
        n_envs=N_ENVS, start_worker_id=START_WORKER, base_port=BASE_PORT,
        learn_id=init["learn_id"], latest_snapshot=latest_snapshot, pool_paths=pool_snapshots,
        weak_p_idle=init["weak_p_idle"],
        opp_probs=opponent_probs_for_phase(1), mate_probs=TEAMMATE_BASE_PROBS,
        file_name=FILE_NAME, seed=SEED, time_scale=TIME_SCALE,
        flatten_for_dqn=FLATTEN_FOR_DQN
    )

    model = DQN("MlpPolicy", env, **DQN_KW)

    eval_env_b0 = build_eval_env(
        start_worker_id=START_WORKER + 999, base_port=BASE_PORT,
        learn_id=0, latest_snapshot=latest_snapshot, pool_paths=pool_snapshots,
        weak_p_idle=0.95, opp_probs={"latest": 0.10, "pool": 0.10, "weak": 0.80}, mate_probs=TEAMMATE_BASE_PROBS,
        file_name=FILE_NAME, seed=SEED+111, time_scale=TIME_SCALE,
        flatten_for_dqn=FLATTEN_FOR_DQN
    )
    eval_env_b1 = build_eval_env(
        start_worker_id=START_WORKER + 1000, base_port=BASE_PORT,
        learn_id=1, latest_snapshot=latest_snapshot, pool_paths=pool_snapshots,
        weak_p_idle=0.95, opp_probs={"latest": 0.10, "pool": 0.10, "weak": 0.80}, mate_probs=TEAMMATE_BASE_PROBS,
        file_name=FILE_NAME, seed=SEED+222, time_scale=TIME_SCALE,
        flatten_for_dqn=FLATTEN_FOR_DQN
    )

    eval_cb_b0 = EvalCallback(
        eval_env=eval_env_b0, best_model_save_path=os.path.join(BEST_DIR, "b0"),
        log_path=os.path.join(EVAL_LOG, "b0"),
        eval_freq=120_000, n_eval_episodes=8, deterministic=True, render=False
    )
    eval_cb_b1 = EvalCallback(
        eval_env=eval_env_b1, best_model_save_path=os.path.join(BEST_DIR, "b1"),
        log_path=os.path.join(EVAL_LOG, "b1"),
        eval_freq=120_000, n_eval_episodes=8, deterministic=True, render=False
    )

    # ===== 엔트로피 스케줄 (DQN에는 효과 없음; 호환성용 no-op) =====
    def set_ent_for_phase(phase_idx: int):
        # DQN은 ent_coef 미사용. 유지해도 무해(무시됨).
        pass

    # ===== 학습 루프 (6h) =====
    for phase_idx, ph in enumerate(CURRICULUM, start=1):
        learn_id = ph["learn_id"]
        weak_p = ph["weak_p_idle"]
        minutes = ph["minutes"]

        # 스냅샷 갱신
        all_snaps = list_snapshots(SNAP_DIR, "*.zip")
        latest_snapshot = all_snaps[-1] if all_snaps else None
        pool_snapshots = all_snaps[:-1][-4:] if len(all_snaps) > 1 else []

        OPP_PROBS = opponent_probs_for_phase(phase_idx)

        # ----- 환경 재빌드 -----
        try:
            env.close()
        except Exception:
            pass
        del env
        gc.collect()
        time.sleep(0.8)

        env = build_vec_env(
            n_envs=N_ENVS,
            start_worker_id=START_WORKER + phase_idx * 24,
            base_port=BASE_PORT,
            learn_id=learn_id,
            latest_snapshot=latest_snapshot,
            pool_paths=pool_snapshots,
            weak_p_idle=weak_p,
            opp_probs=OPP_PROBS,
            mate_probs=TEAMMATE_BASE_PROBS,
            file_name=FILE_NAME,
            seed=SEED + phase_idx * 123,
            time_scale=TIME_SCALE,
            flatten_for_dqn=FLATTEN_FOR_DQN
        )
        model.set_env(env)

        set_ent_for_phase(phase_idx)

        print(f"\n[PHASE] {phase_idx}/{len(CURRICULUM)}  learn_id={learn_id}  weak_p={weak_p:.3f}  minutes={minutes}  opp_probs={OPP_PROBS}")
        t0 = time.time()
        while time.time() - t0 < minutes * 60:
            model.learn(total_timesteps=120_000, reset_num_timesteps=False, callback=[eval_cb_b0, eval_cb_b1])
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{now}] phase {phase_idx} elapsed={(time.time()-t0)/60:.1f} min")

        # ----- 페이즈 종료: 최신 모델 저장(갱신) -----
        try:
            atomic_save_model(model, FINAL_LATEST)
            print(f"[LATEST SAVED] {FINAL_LATEST}.zip (after phase {phase_idx})")
        except Exception as e:
            print(f"[WARN] latest save failed at phase {phase_idx}: {repr(e)}")

        # 스냅샷 저장(짝수 페이즈만)
        if phase_idx % 2 == 0:
            os.makedirs(SNAP_DIR, exist_ok=True)
            stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            snap_path = os.path.join(SNAP_DIR, f"dqn_{stamp}")
            model.save(snap_path)
            print("[SNAPSHOT SAVED]", snap_path + ".zip")

    # ----- 종료 정리 -----
    try:
        env.close()
    except Exception:
        pass
    try:
        eval_env_b0.close()
        eval_env_b1.close()
    except Exception:
        pass

    # 최종 저장
    try:
        atomic_save_model(model, FINAL_LATEST)
        print(f"[SAVE-LAST] {FINAL_LATEST}.zip")
    except Exception as e:
        print(f"[WARN] final latest save failed: {repr(e)}")

    b0_cand = os.path.join(BEST_DIR, "b0", "best_model.zip")
    b1_cand = os.path.join(BEST_DIR, "b1", "best_model.zip")
    if os.path.exists(b0_cand):
        print(f"[BEST-b0] {b0_cand}")
    if os.path.exists(b1_cand):
        print(f"[BEST-b1] {b1_cand}")
    print("[DONE] Dual DQN Self-Play (6h, BLUE, SAFE) finished.")
