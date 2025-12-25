# -*- coding: utf-8 -*-
# file: soccer_demo_q_vs_ppo.py
# 목적: Q-러닝(Q-table, .pkl) vs PPO(.zip) 대결 데모 (watch 모드)
# - BLUE: Q-table, ORANGE: PPO (기본). 한 줄로 서로 뒤바꿀 수 있음.
# - 관측 336/338 자동 보정(one-hot ±2)
# - Unity 타임아웃/포트 충돌 자동 재시도
# - 행동공간 변환기 내장: Discrete(27) ↔ MultiDiscrete(nvec) 자동 변환

import os, time, random, gc, pickle, warnings
import numpy as np
import gym
from gym import spaces
from stable_baselines3 import PPO
import soccer_twos
from mlagents_envs.exception import UnityTimeOutException
warnings.filterwarnings("ignore", category=FutureWarning)

# =========================
# 구성: 누가 Q-table이고 누가 PPO인가?
# =========================
BLUE_SIDE   = "q"     # "q" or "ppo"
ORANGE_SIDE = "ppo"   # "q" or "ppo"

# =========================
# 파일 경로
# =========================
# Q-table (공유/개별 지원)
QTABLE_SHARED = "../qtables/final_q_table.pkl"
QTABLE_BLUE_B0 = "../qtables/blue_b0.pkl"
QTABLE_BLUE_B1 = "../qtables/blue_b1.pkl"
QTABLE_ORNG_O2 = "../qtables/orange_o2.pkl"
QTABLE_ORNG_O3 = "../qtables/orange_o3.pkl"

# PPO
PPO_BLUE_B0    = "../PPO/models_dual_final_blue/dualppo_latest.zip"
PPO_BLUE_B1    = "../PPO/models_dual_final_blue/dualppo_latest.zip"
PPO_ORNG_O2    = "../PPO/models_dual_final_orange/dualppo_latest.zip"
PPO_ORNG_O3    = "../PPO/models_dual_final_orange/dualppo_latest.zip"

# =========================
# Unity/환경 세팅
# =========================
BASE_PORT   = 9700
WORKER_ID   = 2468
TIME_SCALE  = 1.0
STEPS       = 5000

# =========================
# Q-table 유틸 (초심자용 Q-러닝 코드와 호환)
# =========================
K_CAND = 8

def discretize_obs(obs: np.ndarray, decimals: int = 1, clip: float = 5.0):
    x = np.asarray(obs, dtype=np.float32)
    if np.isfinite(clip):
        x = np.clip(x, -clip, clip)
    return tuple(np.round(x, decimals=decimals).tolist())

def sample_action_candidates(action_space: gym.Space, rng, k=K_CAND):
    assert isinstance(action_space, spaces.Discrete) or isinstance(action_space, spaces.MultiDiscrete)
    # 후보는 데모 action_space 기준으로 뽑지 않고, QPolicy 내부에서 Discrete 기준만 사용
    if isinstance(action_space, spaces.Discrete):
        cands = [0]
        for _ in range(max(1, k-1)):
            cands.append(int(action_space.sample()))
        return cands
    else:
        # MultiDiscrete인 경우에도 임시로 0~(총조합-1) 후보 생성은 무의미하므로
        # QPolicy는 Discrete 환경에서만 쓰게 설계 → 여기선 사용 안 함
        return [0]

class QTablePolicy:
    """
    pickle로 저장된 Q-table(dict[(state_key, action)] = Q)을 SB3 정책처럼 감싸는 래퍼
    - 상태키 2종 모두 지원:
      (A) (state_tuple, action)
      (B) ((agent_id,)+state_tuple, action)
    """
    def __init__(self, pkl_path, agent_id=None, seed=0, decimals=1, clip=5.0):
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"Q-table pkl not found: {pkl_path}")
        with open(pkl_path, "rb") as f:
            table = pickle.load(f)
        if not isinstance(table, dict):
            raise ValueError(f"Invalid Q-table format in {pkl_path}: expected dict, got {type(table)}")
        self.values = table
        self.agent_id = agent_id
        self.rng = np.random.default_rng(seed)
        self.decimals = decimals
        self.clip = clip
        self.action_space = None  # env 생성 후 주입 (Discrete(27) 예상)

    def set_action_space(self, action_space):
        self.action_space = action_space

    def _ak(self, a):
        return int(a)

    def _state_keys(self, obs_tuple):
        keys = [("plain", obs_tuple)]
        if self.agent_id is not None:
            keys.append(("with_id", (int(self.agent_id),) + obs_tuple))
        return keys

    def _best_action(self, obs):
        if self.action_space is None:
            raise RuntimeError("Action space not set. Call set_action_space() after env is created.")
        assert isinstance(self.action_space, spaces.Discrete), "QTablePolicy는 Discrete action에서만 사용하세요."
        obs_tuple = discretize_obs(obs, self.decimals, self.clip)
        cands = [0]
        for _ in range(max(1, K_CAND-1)):
            cands.append(int(self.action_space.sample()))

        best_v, best_a = None, None
        # (A) → (B) 순서
        for _, s_key in self._state_keys(obs_tuple):
            found_any = False
            local_best_v, local_best_a = None, None
            for a in cands:
                v = self.values.get((s_key, self._ak(a)))
                if v is None:
                    continue
                found_any = True
                if (local_best_v is None) or (v > local_best_v):
                    local_best_v, local_best_a = v, a
            if found_any:
                best_v, best_a = local_best_v, local_best_a
                break

        if best_a is None:
            best_a = cands[self.rng.integers(low=0, high=len(cands))]
        return int(best_a)

    def predict(self, obs, deterministic=True):
        a = self._best_action(obs)
        return int(a), None

# =========================
# 관측차원(336/338) 보정 유틸
# =========================
def get_expected_dim(model):
    try:
        if hasattr(model, "observation_space") and model.observation_space is not None:
            shp = model.observation_space.shape
            if shp: return int(np.prod(shp))
    except Exception:
        pass
    try:
        pol = getattr(model, "policy", None)
        if pol is not None and hasattr(pol, "observation_space") and pol.observation_space is not None:
            shp = pol.observation_space.shape
            if shp: return int(np.prod(shp))
    except Exception:
        pass
    return None

def with_onehot(x: np.ndarray, which: int):
    oh = np.array([1.0, 0.0], dtype=np.float32) if which == 0 else np.array([0.0, 1.0], dtype=np.float32)
    return np.concatenate([x.astype(np.float32, copy=False), oh], axis=0)

def fix_obs_for_model(model, obs_vec: np.ndarray, default_onehot: int):
    x = np.asarray(obs_vec, dtype=np.float32)
    exp_dim = get_expected_dim(model)
    if exp_dim is None:
        return with_onehot(x, default_onehot) if x.shape[0] == 336 else x
    if x.shape[0] == exp_dim:
        return x
    if x.shape[0] + 2 == exp_dim:
        return with_onehot(x, default_onehot)
    if x.shape[0] - 2 == exp_dim:
        return x[:exp_dim]
    if x.shape[0] < exp_dim:
        pad = np.zeros(exp_dim - x.shape[0], dtype=np.float32)
        return np.concatenate([x, pad], axis=0)
    return x[:exp_dim]

# =========================
# 행동공간 변환기 (핵심 수정)
# =========================
def convert_action_for_env(model, a, demo_action_space):
    """
    모델이 예측한 a를 데모 환경의 action_space 형태로 변환.
    - 모델 action_space 추출 가능 시 이를 기준으로 변환
    - 예:
      모델 MultiDiscrete -> 데모 Discrete : ravel_multi_index로 평탄화
      모델 Discrete      -> 데모 MultiDiscrete : unravel로 분해
      Discrete↔Discrete, MultiDiscrete↔MultiDiscrete : 형태 맞게 캐스팅
    """
    # 모델이 학습 당시의 action_space를 보유하는 경우가 많음
    model_act_space = getattr(model, "action_space", None)

    # 데모가 Discrete
    if isinstance(demo_action_space, spaces.Discrete):
        # (1) 모델이 MultiDiscrete인 경우: 분기형 → 단일 인덱스
        if isinstance(model_act_space, spaces.MultiDiscrete) or (isinstance(a, (list, np.ndarray)) and np.size(a) > 1):
            # a를 1D int array로
            a_arr = np.asarray(a, dtype=np.int64).reshape(-1)
            if model_act_space is not None and isinstance(model_act_space, spaces.MultiDiscrete):
                nvec = tuple(int(x) for x in model_act_space.nvec)
            else:
                # 안전 기본값(소커투스 기본 3-branch 예): 추정이 필요하면 27=3*3*3으로 가정
                # 하지만 가지 수 모를 땐 ravel_multi_index 못 쓰므로 가장 왼쪽 3개만 사용
                # 여기서는 보수적으로 3-branch 가정
                nvec = (3, 3, 3)
            # 부족/초과 길이 방지
            if len(a_arr) != len(nvec):
                # 길이가 다르면 앞쪽부터 맞춰주거나 잘라내기
                m = min(len(a_arr), len(nvec))
                a_arr = a_arr[:m]
                nvec = nvec[:m]
                # 길이 1이면 결국 그냥 그 값만 사용
                if m == 1:
                    return int(a_arr[0])
            try:
                flat = int(np.ravel_multi_index(tuple(a_arr), dims=nvec))
                return flat
            except Exception:
                # 실패 시 첫 요소만 사용
                return int(a_arr[0])

        # (2) 모델이 Discrete, 또는 길이1 배열
        if np.isscalar(a):
            return int(a)
        a_arr = np.asarray(a).reshape(-1)
        return int(a_arr[0])

    # 데모가 MultiDiscrete
    elif isinstance(demo_action_space, spaces.MultiDiscrete):
        # (1) 모델이 Discrete: 단일 인덱스를 분해
        if np.isscalar(a) or (isinstance(a, (list, np.ndarray)) and np.size(a) == 1):
            idx = int(np.asarray(a).reshape(-1)[0])
            nvec = tuple(int(x) for x in demo_action_space.nvec)
            try:
                branch = np.array(np.unravel_index(idx, nvec), dtype=np.int64)
                return branch
            except Exception:
                # 실패 시 전부 0
                return np.zeros_like(demo_action_space.nvec, dtype=np.int64)

        # (2) 모델도 MultiDiscrete라면 형태만 맞춰 반환
        a_arr = np.asarray(a, dtype=np.int64).reshape(-1)
        # 길이 안 맞으면 잘라/패드
        m = len(demo_action_space.nvec)
        if a_arr.size < m:
            a_arr = np.pad(a_arr, (0, m - a_arr.size), constant_values=0)
        elif a_arr.size > m:
            a_arr = a_arr[:m]
        return a_arr

    else:
        # 다른 공간은 여기서 다루지 않음
        if np.isscalar(a):
            return a
        a_arr = np.asarray(a).reshape(-1)
        return a_arr[0] if a_arr.size else 0

# =========================
# Unity 데모 환경 실행 유틸
# =========================
def make_demo_env_with_retries(
    file_name=None, base_port=BASE_PORT, worker_id=WORKER_ID, time_scale=TIME_SCALE,
    blue_team_name="BLUE", orange_team_name="ORANGE",
    max_retries=3, timeout_wait=180
):
    for attempt in range(1, max_retries + 1):
        try:
            print(f"[DEMO] Launch attempt {attempt}: worker_id={worker_id}, base_port={base_port}")
            env = soccer_twos.make(
                watch=True,
                time_scale=time_scale,
                file_name=file_name,
                worker_id=worker_id,
                base_port=base_port,
                blue_team_name=blue_team_name,
                orange_team_name=orange_team_name,
                timeout_wait=timeout_wait,
                flatten_branched=True,  # Discrete(27)로 단순화
            )
            return env
        except UnityTimeOutException as e:
            print(f"[WARN] UnityTimeOutException: {e}")
            time.sleep(2.0); gc.collect()
            worker_id += 1; base_port += random.randint(5, 25)
    raise RuntimeError("Failed to launch Unity demo after retries.")

# =========================
# 로더
# =========================
def load_q_policy_for_agent(default_shared_path, specific_path=None, agent_id=None, side_name=""):
    use_path = None
    if specific_path and os.path.exists(specific_path):
        use_path = specific_path
    elif default_shared_path and os.path.exists(default_shared_path):
        use_path = default_shared_path

    if use_path is None:
        raise FileNotFoundError(
            f"[LOAD][{side_name}] Q-table not found.\n"
            f"  specific: {specific_path}\n"
            f"  shared  : {default_shared_path}"
        )
    print(f"[LOAD][{side_name}] Q-table: {use_path} (agent_id={agent_id})")
    return QTablePolicy(use_path, agent_id=agent_id, seed=0, decimals=1, clip=5.0)

def load_policy(kind: str, path_zip: str, default_shared_q: str, specific_q: str, agent_id=None, side_name=""):
    if kind == "ppo":
        if not os.path.exists(path_zip):
            raise FileNotFoundError(f"[LOAD][{side_name}] PPO zip not found: {path_zip}")
        print(f"[LOAD][{side_name}] PPO: {path_zip}")
        model = PPO.load(path_zip, env=None, device="cpu")
        return model
    elif kind == "q":
        return load_q_policy_for_agent(default_shared_q, specific_q, agent_id=agent_id, side_name=side_name)
    else:
        raise ValueError("kind must be 'ppo' or 'q'")

# =========================
# 메인
# =========================
def main():
    # --- 각 에이전트별 모델 준비 ---
    if BLUE_SIDE == "q":
        b0 = load_policy("q",  PPO_BLUE_B0,  QTABLE_SHARED, QTABLE_BLUE_B0, agent_id=0, side_name="BLUE-b0")
        b1 = load_policy("q",  PPO_BLUE_B1,  QTABLE_SHARED, QTABLE_BLUE_B1, agent_id=1, side_name="BLUE-b1")
    else:
        b0 = load_policy("ppo", PPO_BLUE_B0,  QTABLE_SHARED, QTABLE_BLUE_B0, side_name="BLUE-b0")
        b1 = load_policy("ppo", PPO_BLUE_B1,  QTABLE_SHARED, QTABLE_BLUE_B1, side_name="BLUE-b1")

    if ORANGE_SIDE == "q":
        o2 = load_policy("q",  PPO_ORNG_O2, QTABLE_SHARED, QTABLE_ORNG_O2, agent_id=2, side_name="ORANGE-o2")
        o3 = load_policy("q",  PPO_ORNG_O3, QTABLE_SHARED, QTABLE_ORNG_O3, agent_id=3, side_name="ORANGE-o3")
    else:
        o2 = load_policy("ppo", PPO_ORNG_O2, QTABLE_SHARED, QTABLE_ORNG_O2, side_name="ORANGE-o2")
        o3 = load_policy("ppo", PPO_ORNG_O3, QTABLE_SHARED, QTABLE_ORNG_O3, side_name="ORANGE-o3")

    # --- Unity 실행 ---
    demo = make_demo_env_with_retries(
        file_name=None,
        base_port=BASE_PORT,
        worker_id=WORKER_ID,
        time_scale=TIME_SCALE,
        blue_team_name=f"BLUE ({'Q' if BLUE_SIDE=='q' else 'PPO'})",
        orange_team_name=f"ORANGE ({'Q' if ORANGE_SIDE=='q' else 'PPO'})"
    )

    # Q-table 폴리시에 action_space 주입(필수: Discrete(27) 가정)
    for pol in (b0, b1, o2, o3):
        if isinstance(pol, QTablePolicy):
            pol.set_action_space(demo.action_space)

    # --- 시뮬레이션 ---
    obs = demo.reset()
    print(f"[RUN] Start demo: steps={STEPS}, env_action_space={demo.action_space}")
    for t in range(STEPS):
        def act(model, obs_vec, default_onehot):
            # PPO는 관측차원 보정 필요
            if isinstance(model, PPO):
                x = fix_obs_for_model(model, np.asarray(obs_vec, dtype=np.float32), default_onehot)
                a, _ = model.predict(x, deterministic=True)
                return convert_action_for_env(model, a, demo.action_space)
            else:
                # QTablePolicy: 버킷팅 내부 처리
                a, _ = model.predict(np.asarray(obs_vec, dtype=np.float32), deterministic=True)
                return convert_action_for_env(model, a, demo.action_space)

        a0 = act(b0, obs[0], 0)  # BLUE b0
        a1 = act(b1, obs[1], 1)  # BLUE b1
        a2 = act(o2, obs[2], 0)  # ORANGE o2
        a3 = act(o3, obs[3], 1)  # ORANGE o3

        obs, rew, done, info = demo.step({0:a0, 1:a1, 2:a2, 3:a3})
        if done.get("__all__", False):
            obs = demo.reset()

    demo.close()
    print("[DEMO] finished.")

if __name__ == "__main__":
    main()
