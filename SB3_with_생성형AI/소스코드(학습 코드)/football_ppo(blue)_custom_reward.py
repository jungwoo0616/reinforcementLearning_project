# -*- coding: utf-8 -*-
# file: soccer_train_dual_selfplay_6h_blue_safe.py
# 목적: Dual PPO Self-Play (BLUE 0/1) - 6시간 커리큘럼(안정성 강화: WinError1455 및 CUDA 초기화/포트 충돌 회피)
# - 핵심: base_port를 두 번 더하던 버그 제거(내부에서 base_port + worker_id 계산)
# - DummyVecEnv(단일 프로세스), SB3/torch __main__ 임포트, 스냅샷 CPU 로드, 관측차원 안전보정
# - 환경 재빌드 시 graceful close + GC + 짧은 대기, 바인딩 실패/worker 충돌 시 재시도(backoff)
# - ★ 매 페이즈 종료 시 최신 모델을 models_dual_final_blue/dualppo_latest.zip 로 저장(갱신)
# - ★ 커스텀 보상: 패스 보상(pass_reward) + 점유율 보상(possession_reward) 추가(BLUE 기준)

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
                return self._with_onehot(x, which=0)  # 336 -> 338 (team-internal one-hot)
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

# ------------------ Dual Self-Play 학습 환경 (BLUE 0/1 학습자) ------------------
class SelfPlaySoccerEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self,
                 learn_agent_id: int,           # 0 또는 1
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
                 # ==== 커스텀 보상 파라미터 ====
                 pass_reward: float = 0.05,        # 같은 팀 패스 보상
                 possession_reward: float = 0.005  # BLUE가 공을 소유한 스텝당 보상
                 ):
        super().__init__()
        assert learn_agent_id in (0, 1)   # ★ BLUE 팀
        self.learn_id = int(learn_agent_id)

        # 중요: base_port는 그대로, worker_id만 전달 (내부에서 base_port + worker_id)
        self._env = soccer_twos.make(
            render=render, time_scale=time_scale, file_name=file_name,
            worker_id=worker_id, base_port=base_port
        )

        self.orig_obs_space: spaces.Box = self._env.observation_space
        self.action_space: spaces.Space = self._env.action_space

        # 팀 내부 식별(one-hot 2칸: BLUE 0 vs BLUE 1)
        low = np.full(self.orig_obs_space.shape[0] + 2, -np.inf, dtype=np.float32)
        high = np.full(self.orig_obs_space.shape[0] + 2, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        opponent_probs = opponent_probs or {"latest": 0.0, "pool": 0.0, "weak": 1.0}
        teammate_probs = teammate_probs or {"latest": 0.60, "pool": 0.30, "weak": 0.10}

        self._base_weak_p = float(weak_p_idle)

        # BLUE 팀 동료(0/1 중 다른 한 명)
        self._teammate = PolicyMixer(self.action_space, weak_p_idle=weak_p_idle,
                                     latest_path=latest_snapshot, pool_paths=pool_snapshots,
                                     probs=teammate_probs, seed=seed+1)
        # ORANGE 팀 상대(2,3)
        self._opponent0 = PolicyMixer(self.action_space, weak_p_idle=weak_p_idle,
                                      latest_path=latest_snapshot, pool_paths=pool_snapshots,
                                      probs=opponent_probs, seed=seed+2)
        self._opponent1 = PolicyMixer(self.action_space, weak_p_idle=weak_p_idle,
                                      latest_path=latest_snapshot, pool_paths=pool_snapshots,
                                      probs=opponent_probs, seed=seed+3)

        self._obs_dict = None
        self.rng = np.random.default_rng(seed)

        # ===== Reward Shaping 상태 =====
        # 환경 info에서 읽어올 것으로 가정: ball_owned_team, ball_owned_player
        # 예: ball_owned_team: -1(없음), 0(BLUE), 1(ORANGE)
        self._last_ball_team = None
        self._last_ball_player = None
        self._BLUE_TEAM_IDX = 0

        self.pass_reward = float(pass_reward)
        self.possession_reward = float(possession_reward)

    def _aug_obs(self, obs_vec: np.ndarray) -> np.ndarray:
        # BLUE 0 -> [1,0], BLUE 1 -> [0,1]
        onehot = np.array([1.0, 0.0], dtype=np.float32) if self.learn_id == 0 else np.array([0.0, 1.0], dtype=np.float32)
        return np.concatenate([obs_vec.astype(np.float32, copy=False), onehot], axis=0)

    def _resample_difficulty(self):
        sampled = float(np.clip(np.random.normal(self._base_weak_p, 0.01), 0.88, 0.995))
        self._teammate.weak.p_idle = sampled
        self._opponent0.weak.p_idle = sampled
        self._opponent1.weak.p_idle = sampled

    # ==== Reward Shaping 관련 헬퍼 ====
    def _extract_ball_owner(self, info_d):
        """
        info_d[self.learn_id] 안에서 공 소유 정보를 추출.
        환경에 따라 키 이름이 다를 수 있으므로, 실제 info 출력 보고 맞춰서 수정 필요할 수 있음.
        """
        info = info_d.get(self.learn_id, {})
        team = info.get("ball_owned_team", None)
        player = info.get("ball_owned_player", None)
        return team, player

    def _compute_shaping_reward(self, info_d) -> float:
        """
        - BLUE가 공을 소유한 스텝마다 작은 보상(점유율)
        - 같은 팀(BLUE) 내에서 플레이어가 바뀌면 패스로 간주하고 추가 보상
        """
        shaped = 0.0
        team, player = self._extract_ball_owner(info_d)

        # info에 공 소유 정보가 없으면 shaping 안 함
        if team is None or player is None:
            return 0.0

        # 1) 점유율 보상: BLUE 팀이 공을 가지고 있으면 매 스텝마다 +possession_reward
        if team == self._BLUE_TEAM_IDX:
            shaped += self.possession_reward

        # 2) 패스 보상: 직전 소유자와 현재 소유자가 같은 팀(BLUE)이면서, 서로 다른 플레이어일 때
        if (self._last_ball_team == self._BLUE_TEAM_IDX and
            team == self._BLUE_TEAM_IDX and
            self._last_ball_player is not None and
            player is not None and
            player != self._last_ball_player):
            shaped += self.pass_reward

        # 상태 업데이트
        self._last_ball_team = team
        self._last_ball_player = player

        return shaped
    # ==== Reward Shaping 헬퍼 끝 ====

    def reset(self):
        self._resample_difficulty()
        self._obs_dict = self._env.reset()

        # ball owner 상태 초기화
        self._last_ball_team = None
        self._last_ball_player = None

        o = np.asarray(self._obs_dict[self.learn_id], dtype=np.float32)
        return self._aug_obs(o)

    def step(self, action):
        if isinstance(self.action_space, spaces.Discrete):
            aL = int(action)
        elif isinstance(self.action_space, spaces.MultiDiscrete):
            aL = np.asarray(action, dtype=np.int64)
        else:
            raise TypeError(f"Unsupported action_space: {type(self.action_space)}")

        # BLUE 팀(0,1) 학습자/동료 배치
        if self.learn_id == 0:
            a0 = aL
            a1 = self._teammate.act(np.asarray(self._obs_dict[1], dtype=np.float32))
        else:  # learn_id == 1
            a0 = self._teammate.act(np.asarray(self._obs_dict[0], dtype=np.float32))
            a1 = aL

        # ORANGE 팀 상대(2,3)
        a2 = self._opponent0.act(np.asarray(self._obs_dict[2], dtype=np.float32))
        a3 = self._opponent1.act(np.asarray(self._obs_dict[3], dtype=np.float32))

        next_obs_d, rew_d, done_d, info_d = self._env.step({0: a0, 1: a1, 2: a2, 3: a3})
        self._obs_dict = next_obs_d

        obs_sel = np.asarray(next_obs_d[self.learn_id], dtype=np.float32)

        # ----- 원래 환경 보상 + shaping 보상 합치기 -----
        base_rew = float(rew_d[self.learn_id])
        shaping = self._compute_shaping_reward(info_d)
        rew_sel = base_rew + shaping

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
    """
    SB3의 model.save는 'path'에 저장하고 '.zip'을 자동 부착.
    원자적 저장을 위해 임시 경로에 먼저 저장 후 os.replace로 교체.
    """
    os.makedirs(os.path.dirname(target_path_without_ext), exist_ok=True)
    tmp_base = target_path_without_ext + "_tmp"
    final_zip = target_path_without_ext + ".zip"
    tmp_zip = tmp_base + ".zip"
    try:
        model.save(tmp_base)          # -> tmp_base.zip 생성
        if os.path.exists(final_zip):
            os.replace(tmp_zip, final_zip)  # 원자적 교체
        else:
            os.rename(tmp_zip, final_zip)
    finally:
        if os.path.exists(tmp_zip):
            try: os.remove(tmp_zip)
            except Exception: pass

# ------------------ 메인 ------------------
if __name__ == "__main__":
    from stable_baselines3 import PPO
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
    BASE_PORT = 9200
    START_WORKER = 520
    N_ENVS = 6
    TIME_SCALE = 20.0

    # ★ 커스텀 보상 계수 (추천 스케일)
    PASS_REWARD = 0.05        # 같은 팀 패스 보상
    POSSESSION_REWARD = 0.005 # BLUE가 공을 소유한 스텝당 보상

    # ★ 블루 버전 저장 경로/이름
    SNAP_DIR = "snapshots_dual_blue"
    BEST_DIR = "best_models_dual_blue"
    EVAL_LOG = "eval_logs_dual_blue"
    FINAL_DIR = "models_dual_final_blue"   # 최신 모델 저장 디렉터리
    FINAL_LATEST = os.path.join(FINAL_DIR, "dualppo_latest")  # dualppo_latest.zip 로 갱신

    os.makedirs(SNAP_DIR, exist_ok=True)
    os.makedirs(os.path.join(BEST_DIR, "b0"), exist_ok=True)   # BLUE-0 best
    os.makedirs(os.path.join(BEST_DIR, "b1"), exist_ok=True)   # BLUE-1 best
    os.makedirs(os.path.join(EVAL_LOG, "b0"), exist_ok=True)
    os.makedirs(os.path.join(EVAL_LOG, "b1"), exist_ok=True)
    os.makedirs(FINAL_DIR, exist_ok=True)

    # ===== 6시간 커리큘럼 ===== (오렌지와 대칭: 0/1)
    CURRICULUM = [
        # Warmup (40)
        {"minutes": 8, "learn_id": 0, "weak_p_idle": 0.995},
        {"minutes": 8, "learn_id": 1, "weak_p_idle": 0.992},
        {"minutes": 8, "learn_id": 0, "weak_p_idle": 0.990},
        {"minutes": 8, "learn_id": 1, "weak_p_idle": 0.988},
        {"minutes": 8, "learn_id": 0, "weak_p_idle": 0.985},
        # Mid (200)
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
        # Sparring (60)
        {"minutes": 10, "learn_id": 0, "weak_p_idle": 0.955},
        {"minutes": 10, "learn_id": 1, "weak_p_idle": 0.955},
        {"minutes": 10, "learn_id": 0, "weak_p_idle": 0.950},
        {"minutes": 10, "learn_id": 1, "weak_p_idle": 0.950},
        {"minutes": 10, "learn_id": 0, "weak_p_idle": 0.948},
        {"minutes": 10, "learn_id": 1, "weak_p_idle": 0.948},
        # Finish (60) — 0 비중 소폭↑
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
                    render, time_scale, file_name, seed, worker_id, base_port,
                    pass_reward, possession_reward):
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
                base_port=base_port,  # base_port 그대로
                pass_reward=pass_reward,
                possession_reward=possession_reward
            )
        return _init

    def _build_vec_env_with_retry(n_envs, start_worker_id, base_port,
                                  learn_id, latest_snapshot, pool_paths, weak_p_idle, opp_probs, mate_probs,
                                  file_name, seed, time_scale=20.0, max_attempts=4,
                                  pass_reward=PASS_REWARD, possession_reward=POSSESSION_REWARD):
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
                                    False, time_scale, file_name, seed + i, wid, base_port,
                                    pass_reward, possession_reward)
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
                      file_name, seed, time_scale=20.0,
                      pass_reward=PASS_REWARD, possession_reward=POSSESSION_REWARD):
        return _build_vec_env_with_retry(
            n_envs, start_worker_id, base_port,
            learn_id, latest_snapshot, pool_paths, weak_p_idle, opp_probs, mate_probs,
            file_name, seed, time_scale,
            pass_reward=pass_reward, possession_reward=possession_reward
        )

    def build_eval_env(start_worker_id, base_port,
                       learn_id, latest_snapshot, pool_paths, weak_p_idle, opp_probs, mate_probs,
                       file_name, seed, time_scale=20.0,
                       pass_reward=PASS_REWARD, possession_reward=POSSESSION_REWARD):
        from stable_baselines3.common.vec_env import DummyVecEnv
        wid = start_worker_id
        return DummyVecEnv([
            make_env_fn(learn_id, latest_snapshot, pool_paths, weak_p_idle, opp_probs, mate_probs,
                        False, time_scale, file_name, seed, wid, base_port,
                        pass_reward, possession_reward)
        ])

    # ===== PPO 하이퍼 =====
    def lr_schedule(progress_remaining: float):
        return 1e-4 + (3e-4 - 1e-4) * progress_remaining

    PPO_KW = dict(
        n_steps=1024,
        batch_size=2048,
        learning_rate=lr_schedule,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        policy_kwargs=dict(net_arch=[512, 256]),
        verbose=1,
        seed=SEED,
        device=DEVICE,
        tensorboard_log="runs/ppo_dual_selfplay_6h_blue",
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
        pass_reward=PASS_REWARD, possession_reward=POSSESSION_REWARD
    )

    model = PPO("MlpPolicy", env, **PPO_KW)

    # BLUE 0/1 평가 환경
    eval_env_b0 = build_eval_env(
        start_worker_id=START_WORKER + 999, base_port=BASE_PORT,
        learn_id=0, latest_snapshot=latest_snapshot, pool_paths=pool_snapshots,
        weak_p_idle=0.95, opp_probs={"latest":0.10,"pool":0.10,"weak":0.80}, mate_probs=TEAMMATE_BASE_PROBS,
        file_name=FILE_NAME, seed=SEED+111, time_scale=TIME_SCALE,
        pass_reward=PASS_REWARD, possession_reward=POSSESSION_REWARD
    )
    eval_env_b1 = build_eval_env(
        start_worker_id=START_WORKER + 1000, base_port=BASE_PORT,
        learn_id=1, latest_snapshot=latest_snapshot, pool_paths=pool_snapshots,
        weak_p_idle=0.95, opp_probs={"latest":0.10,"pool":0.10,"weak":0.80}, mate_probs=TEAMMATE_BASE_PROBS,
        file_name=FILE_NAME, seed=SEED+222, time_scale=TIME_SCALE,
        pass_reward=PASS_REWARD, possession_reward=POSSESSION_REWARD
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

    # ===== 엔트로피 스케줄 =====
    ENT_WARM = 0.010   # 1~5
    ENT_MID1 = 0.009   # 6~10
    ENT_MID2 = 0.008   # 11~15
    ENT_SPAR = 0.007   # 16~21
    ENT_FIN  = 0.006   # 22~25
    def set_ent_for_phase(phase_idx: int):
        if phase_idx <= 5:   model.ent_coef = ENT_WARM
        elif phase_idx <= 10: model.ent_coef = ENT_MID1
        elif phase_idx <= 15: model.ent_coef = ENT_MID2
        elif phase_idx <= 21: model.ent_coef = ENT_SPAR
        else:                 model.ent_coef = ENT_FIN

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

        # ----- 환경 재빌드 (spawn 없음) -----
        try:
            env.close()
        except Exception:
            pass
        del env
        gc.collect()
        time.sleep(0.8)  # 소켓 정리 대기

        env = build_vec_env(
            n_envs=N_ENVS,
            start_worker_id=START_WORKER + phase_idx * 24,  # 블록 간격(24)로 충분히 띄움 (N_ENVS=6)
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
            pass_reward=PASS_REWARD,
            possession_reward=POSSESSION_REWARD
        )
        model.set_env(env)

        # 엔트로피 설정
        set_ent_for_phase(phase_idx)

        print(f"\n[PHASE] {phase_idx}/{len(CURRICULUM)}  learn_id={learn_id}  weak_p={weak_p:.3f}  minutes={minutes}  ent={model.ent_coef}  opp_probs={OPP_PROBS}")
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

        # 스냅샷 저장(짝수 페이즈만 → 파일 과다 방지)
        if phase_idx % 2 == 0:
            os.makedirs(SNAP_DIR, exist_ok=True)
            stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            snap_path = os.path.join(SNAP_DIR, f"dualppo_{stamp}")
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

    # 최종 저장(마지막으로 한 번 더 갱신)
    try:
        atomic_save_model(model, FINAL_LATEST)
        print(f"[SAVE-LAST] {FINAL_LATEST}.zip")
    except Exception as e:
        print(f"[WARN] final latest save failed: {repr(e)}")

    b0_cand = os.path.join(BEST_DIR, "b0", "best_model.zip")
    b1_cand = os.path.join(BEST_DIR, "b1", "best_model.zip")
    if os.path.exists(b0_cand): print(f"[BEST-b0] {b0_cand}")
    if os.path.exists(b1_cand): print(f"[BEST-b1] {b1_cand}")
    print("[DONE] Dual PPO Self-Play (6h, BLUE, SAFE, with pass/possession rewards) finished.")
