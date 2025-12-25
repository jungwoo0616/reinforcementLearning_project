# -*- coding: utf-8 -*-
# file: soccer_demo_best_vs_best.py
# 목적: BLUE(b0,b1,A2C) vs ORANGE(o2,o3,PPO) - 각 팀의 best 모델 4개를 불러와 시청(watch) 데모 실행
# - 관측차원(336/338) 자동 보정: 모델이 기대하는 차원에 맞춰 one-hot(2) 부착/제거
# - Unity 타임아웃/포트 충돌 자동 재시도

import os, time, random, gc
import numpy as np
from gym import spaces
from stable_baselines3 import PPO, A2C   # ← A2C 추가
import soccer_twos
from mlagents_envs.exception import UnityTimeOutException
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------- 경로 설정: 각 팀의 best_model.zip ----------
# ※ 블루팀 경로는 A2C로 학습된 zip로 지정하세요.
BLUE_B0 = os.path.join("..", "A2C", "models_dual_final_blue", "duala2c_latest.zip")
BLUE_B1 = os.path.join("..", "A2C", "models_dual_final_blue", "duala2c_latest.zip")
# ※ 오렌지팀 경로는 PPO로 학습된 zip로 지정하세요.
ORNG_O2 = os.path.join("..", "PPO(vsA2C)", "models_dual_final_orange", "dualppo_latest.zip")
ORNG_O3 = os.path.join("..", "PPO(vsA2C)", "models_dual_final_orange", "dualppo_latest.zip")

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
        [x.astype(np.float32, copy=False),
         np.array([1.0, 0.0], dtype=np.float32) if which == 0 else np.array([0.0, 1.0], dtype=np.float32)],
        axis=0
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

# ---------- 유틸: Unity 데모 환경 안전 실행 ----------
def make_demo_env_with_retries(
    file_name=None,
    base_port=9700,
    worker_id=2468,
    time_scale=1.0,
    blue_team_name="BLUE (A2C: b0,b1)",
    orange_team_name="ORANGE (PPO: o2,o3)",
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
    print("[LOAD] BLUE b0 (A2C):", BLUE_B0)
    print("[LOAD] BLUE b1 (A2C):", BLUE_B1)
    print("[LOAD] ORANGE o2 (PPO):", ORNG_O2)
    print("[LOAD] ORANGE o3 (PPO):", ORNG_O3)

    # 블루팀: A2C로 로드
    b0 = A2C.load(BLUE_B0, env=None, device="cpu")
    b1 = A2C.load(BLUE_B1, env=None, device="cpu")

    # 오렌지팀: PPO로 로드
    o2 = PPO.load(ORNG_O2, env=None, device="cpu")
    o3 = PPO.load(ORNG_O3, env=None, device="cpu")

    # 2) Unity 실행
    demo = make_demo_env_with_retries(
        file_name=None,            # 필요시 절대경로 지정
        base_port=9700,            # 다른 실행과 겹치지 않게
        worker_id=2468,
        time_scale=1.0,            # 시청용
        blue_team_name="BLUE (A2C: b0,b1)",
        orange_team_name="ORANGE (PPO: o2,o3)",
    )

    # 3) 시뮬레이션
    obs = demo.reset()
    STEPS = 5000  # 길이 자유 조정
    for t in range(STEPS):
        # 각 에이전트별 기대 차원에 맞게 관측 보정 후 predict
        a0, _ = b0.predict(fix_obs_for_model(b0, np.asarray(obs[0], dtype=np.float32), default_onehot=0), deterministic=True)
        a1, _ = b1.predict(fix_obs_for_model(b1, np.asarray(obs[1], dtype=np.float32), default_onehot=1), deterministic=True)
        a2, _ = o2.predict(fix_obs_for_model(o2, np.asarray(obs[2], dtype=np.float32), default_onehot=0), deterministic=True)
        a3, _ = o3.predict(fix_obs_for_model(o3, np.asarray(obs[3], dtype=np.float32), default_onehot=1), deterministic=True)

        # 액션 타입 변환
        if isinstance(demo.action_space, spaces.Discrete):
            a0, a1, a2, a3 = int(a0), int(a1), int(a2), int(a3)
        else:
            a0, a1, a2, a3 = map(lambda x: np.array(x, dtype=np.int64), [a0, a1, a2, a3])

        obs, rew, done, info = demo.step({0: a0, 1: a1, 2: a2, 3: a3})

        # 에피소드 종료 시 재시작(화면은 계속)
        if done.get("__all__", False):
            obs = demo.reset()

    demo.close()
    print("[DEMO] finished.")

if __name__ == "__main__":
    main()
