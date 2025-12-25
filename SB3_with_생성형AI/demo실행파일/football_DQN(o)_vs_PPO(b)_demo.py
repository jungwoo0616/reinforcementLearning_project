# -*- coding: utf-8 -*-
# file: soccer_demo_ppo_blue_vs_dqn_orange.py
# 목적: BLUE(b0,b1: PPO) vs ORANGE(o2,o3: DQN) - best 모델 4개를 불러와 시청(watch) 데모 실행
# - 관측차원(336/338) 자동 보정: 모델이 기대하는 차원에 맞춰 one-hot(2) 부착/제거
# - Unity 타임아웃/포트 충돌 자동 재시도
# - ORANGE DQN이 뱉는 Discrete 액션을 MultiDiscrete로 디코딩해서 Unity에 전달

import os, time, random, gc
import numpy as np
from gym import spaces
from stable_baselines3 import PPO, DQN
import soccer_twos
from mlagents_envs.exception import UnityTimeOutException
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------- 경로 설정: 각 팀의 best_model.zip ----------
# 필요에 따라 경로/파일명 수정해서 사용하세요.

# BLUE 팀 PPO
BLUE_B0 = os.path.join("..", "PPO", "models_dual_final_blue",   "dualppo_latest.zip")    # PPO BLUE b0
BLUE_B1 = os.path.join("..", "PPO", "models_dual_final_blue",   "dualppo_latest.zip")    # PPO BLUE b1

# ORANGE 팀 DQN
ORNG_O2 = os.path.join("..", "DQN", "models_dual_final_orange", "dualdqn_latest.zip")    # DQN ORANGE o2
ORNG_O3 = os.path.join("..", "DQN", "models_dual_final_orange", "dualdqn_latest.zip")    # DQN ORANGE o3

# ---------- 유틸: 모델이 기대하는 관측 차원 구하기 ----------
def get_expected_dim(model):
    # SB3는 model.observation_space 또는 model.policy.observation_space 보유 가능
    try:
        if hasattr(model, "observation_space") and model.observation_space is not None:
            shp = model.observation_space.shape
            if shp is not None:
                return int(np.prod(shp))
    except Exception:
        pass
    try:
        if hasattr(model, "policy") and hasattr(model.policy, "observation_space"):
            shp = model.policy.observation_space.shape
            if shp is not None:
                return int(np.prod(shp))
    except Exception:
        pass
    return None  # 못 구하면 None

def with_onehot(x: np.ndarray, which: int):
    # which: 0 -> [1,0], 1 -> [0,1]
    return np.concatenate(
        [
            x.astype(np.float32, copy=False),
            np.array([1.0, 0.0], dtype=np.float32) if which == 0
            else np.array([0.0, 1.0], dtype=np.float32),
        ],
        axis=0,
    )

def fix_obs_for_model(model, obs_vec: np.ndarray, default_onehot: int):
    """
    모델이 기대하는 차원(exp_dim)에 맞춰 obs_vec(336/338)을 보정.
    default_onehot은 해당 모델의 학습 포지션용 one-hot:
      BLUE: b0->0([1,0]), b1->1([0,1])
      ORANGE: o2->0([1,0]), o3->1([0,1])
    """
    x = np.asarray(obs_vec, dtype=np.float32)
    exp_dim = get_expected_dim(model)
    if exp_dim is None:
        # 안전 기본: 336이면 +2 부착(338) 시도
        if x.shape[0] == 336:
            return with_onehot(x, default_onehot)
        return x

    if x.shape[0] == exp_dim:
        return x
    if x.shape[0] + 2 == exp_dim:
        # 모델이 +2(one-hot)를 기대 → 부착
        return with_onehot(x, default_onehot)
    if x.shape[0] - 2 == exp_dim:
        # 입력에 불필요한 2차원 존재 → 마지막 2개 제거
        return x[:exp_dim]

    # 그 외엔 pad/truncate
    if x.shape[0] < exp_dim:
        pad = np.zeros(exp_dim - x.shape[0], dtype=np.float32)
        return np.concatenate([x, pad], axis=0)
    else:
        return x[:exp_dim]

# ---------- DQN Discrete → MultiDiscrete 디코딩 ----------
def dqn_to_multidiscrete(action, action_space: spaces.Space):
    """
    DQN이 뱉는 Discrete 정수(action)를
    Unity가 기대하는 MultiDiscrete 벡터로 변환.
    - MultiDiscrete([n1, n2, ..., nk]) 라고 하면,
      0 <= action < n1*n2*...*nk 라고 가정하고 mixed-radix로 디코딩.
    """
    if not isinstance(action_space, spaces.MultiDiscrete):
        # MultiDiscrete가 아니면 그냥 정수로 반환
        return int(np.asarray(action).item())

    nvec = action_space.nvec
    idx = int(np.asarray(action).item())
    out = np.zeros_like(nvec, dtype=np.int64)

    # 뒤에서부터 나눠가면서 나머지 추출 (mixed radix)
    for i in range(len(nvec) - 1, -1, -1):
        out[i] = idx % nvec[i]
        idx //= nvec[i]

    return out  # shape: (len(nvec),), 예: [0,1,2]

# ---------- 유틸: Unity 데모 환경 안전 실행 ----------
def make_demo_env_with_retries(
    file_name=None,
    base_port=9700,
    worker_id=2468,
    time_scale=1.0,
    blue_team_name="BLUE (PPO b0,b1)",
    orange_team_name="ORANGE (DQN o2,o3)",
    max_retries=3,
    timeout_wait=180
):
    for attempt in range(1, max_retries + 1):
        try:
            print(f"[DEMO] Launch attempt {attempt}: worker_id={worker_id}, base_port={base_port}")
            env = soccer_twos.make(
                watch=True,                  # 화면 보기
                time_scale=time_scale,       # 시청용 1.0 추천
                file_name=file_name,         # Unity 빌드 경로가 있으면 지정, None이면 기본
                worker_id=worker_id,
                base_port=base_port,
                blue_team_name=blue_team_name,
                orange_team_name=orange_team_name,
                timeout_wait=timeout_wait,
            )
            return env
        except UnityTimeOutException as e:
            print(f"[WARN] UnityTimeOutException on attempt {attempt}: {e}")
            time.sleep(3)
            gc.collect()
            worker_id += 1
            base_port += random.randint(5, 25)
    raise RuntimeError("Failed to launch Unity demo after multiple retries.")

def main():
    # 1) 모델 로드
    print("[LOAD] BLUE(PPO) b0:", BLUE_B0)
    print("[LOAD] BLUE(PPO) b1:", BLUE_B1)
    print("[LOAD] ORANGE(DQN) o2:", ORNG_O2)
    print("[LOAD] ORANGE(DQN) o3:", ORNG_O3)

    # BLUE: PPO, ORANGE: DQN
    b0 = PPO.load(BLUE_B0, env=None, device="cpu")
    b1 = PPO.load(BLUE_B1, env=None, device="cpu")
    o2 = DQN.load(ORNG_O2, env=None, device="cpu")
    o3 = DQN.load(ORNG_O3, env=None, device="cpu")

    # 2) Unity 실행
    demo = make_demo_env_with_retries(
        file_name=None,            # 필요시 절대경로 지정
        base_port=9700,            # 다른 실행과 겹치지 않게
        worker_id=2468,
        time_scale=1.0,            # 시청용
        blue_team_name="BLUE PPO (b0,b1)",
        orange_team_name="ORANGE DQN (o2,o3)",
    )

    action_space = demo.action_space

    # 3) 시뮬레이션
    obs = demo.reset()
    STEPS = 5000  # 길이 자유 조정
    for t in range(STEPS):
        # ----- 관측 보정 -----
        obs0 = fix_obs_for_model(b0, np.asarray(obs[0], dtype=np.float32), default_onehot=0)
        obs1 = fix_obs_for_model(b1, np.asarray(obs[1], dtype=np.float32), default_onehot=1)
        obs2 = fix_obs_for_model(o2, np.asarray(obs[2], dtype=np.float32), default_onehot=0)
        obs3 = fix_obs_for_model(o3, np.asarray(obs[3], dtype=np.float32), default_onehot=1)

        # ----- 원시 액션 (모델 출력) -----
        # BLUE: PPO → MultiDiscrete 그대로 사용
        a0_raw, _ = b0.predict(obs0, deterministic=True)
        a1_raw, _ = b1.predict(obs1, deterministic=True)
        # ORANGE: DQN → Discrete action
        a2_raw, _ = o2.predict(obs2, deterministic=True)
        a3_raw, _ = o3.predict(obs3, deterministic=True)

        # ----- 환경 action_space에 맞게 변환 -----
        if isinstance(action_space, spaces.MultiDiscrete):
            # BLUE(PPO): 이미 MultiDiscrete라고 가정 (학습 시와 동일)
            a0 = np.array(a0_raw, dtype=np.int64)
            a1 = np.array(a1_raw, dtype=np.int64)
            # ORANGE(DQN): Discrete -> MultiDiscrete 디코딩
            a2 = dqn_to_multidiscrete(a2_raw, action_space)
            a3 = dqn_to_multidiscrete(a3_raw, action_space)

        elif isinstance(action_space, spaces.Discrete):
            # 완전 Discrete 환경일 경우 (이 케이스는 soccer_twos 기본에선 거의 없음)
            a0 = int(np.asarray(a0_raw).item())
            a1 = int(np.asarray(a1_raw).item())
            a2 = int(np.asarray(a2_raw).item())
            a3 = int(np.asarray(a3_raw).item())

        else:
            # 기타(Box 등) 대비: 그냥 np.array로 캐스팅
            a0 = np.array(a0_raw, dtype=np.int64)
            a1 = np.array(a1_raw, dtype=np.int64)
            a2 = np.array(a2_raw, dtype=np.int64)
            a3 = np.array(a3_raw, dtype=np.int64)

        obs, rew, done, info = demo.step({0: a0, 1: a1, 2: a2, 3: a3})

        # 에피소드 종료 시 재시작(화면은 계속)
        if done.get("__all__", False):
            obs = demo.reset()

    demo.close()
    print("[DEMO] finished.")

if __name__ == "__main__":
    main()
