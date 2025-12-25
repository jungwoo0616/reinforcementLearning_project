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

CURRICULUM_THRESH = -0.16  # 커리큘럼 임계값
SELFPLAY_THRESH   = -0.17  # 셀프플레이 임계값

current = 0 # 현재 커리큘럼

with open("curriculum.yaml", "r", encoding="utf-8") as f:
    curriculum = yaml.load(f, Loader=yaml.FullLoader)
tasks = curriculum["tasks"]

config_fns = {  # 셀프플레이를 하므로 모두 None(랜덤 정책 사용 안함)으로 설정
    "none": lambda *_: None,
    "random_players": lambda env: None,
}

# 매 에피소드마다 에이전트들이 어떤 정책 사용할지 결정하는 함수
def policy_mapping_fn(agent_id, *args, **kwargs):
    if agent_id == 0:  # 에이전트 0번이면
        return "default"  # default 정책 사용
    else:  # 그 외의 에이전트
        return np.random.choice(  # 아래에 있는 정책 중 하나를 확률적으로 선택
            ["default", "opponent_1", "opponent_2", "opponent_3"],
            size=1,  # 이 정책들 중 딱 1개만 선택
            p=[0.50, 0.25, 0.125, 0.125],
        )[0]

class CurriculumSelfPlayUpdateCallback(DefaultCallbacks):
    # 에피소드가 시작될 때마다 RLLib이 자동으로 호출
    def on_episode_start(
        self, *, worker, base_env, policies, episode, env_index, **kwargs
    ) -> None:
        global current, tasks  # 현재 난이도와 커리큘럼
        for env in base_env.get_unwrapped():
            cfg_name = tasks[current]["config_fn"]  # 커리큘럼.yaml에서 현재 난이도에 해당하는 설정 함수 가져옴
            config_fns[cfg_name](env)  # 그 함수를 실행(셀프플레이를 하므로 아무것도 실행 안하긴 함. 신경X)
            env.env_channel.set_parameters(  # 공과 에이전트들의 상태를 강제로 재설정
                ball_state=sample_pos_vel(tasks[current]["ranges"]["ball"]),  # 공의 위치와 속도를 현재 난이도 범위 내에서 랜덤 선택
                players_states={  # 에이전트들의 상태 설정
                    # 에이전트의 위치/회전 정보를 랜덤 선택
                    player: sample_player(tasks[current]["ranges"]["players"][player])
                    # 현재 단계에 정의된 에이전트 목록 만큼 반복
                    for player in tasks[current]["ranges"]["players"]
                },
            )

    # 매 iteration이 끝날 때마다 호출되는 함수
    def on_train_result(self, **info):
        global current, tasks  # 현재 난이도와 커리큘럼

        # 훈련 객체(trainer)와 이번 iteration의 평균 보상 가져옴
        trainer = info["trainer"]
        mean_rew = info["result"]["episode_reward_mean"]

        # 커리큘럼 - 난이도 업데이트
        if mean_rew > CURRICULUM_THRESH:  # 에피소드 평균 보상이 임계값을 넘으면
            if current < len(tasks) - 1:  # 커리큘럼 최고 난이도가 아니라면
                print("---- Updating curriculum task!!! ----")
                current += 1 # 난이도 1단계 증가
                print(f"Current task: {current} - {tasks[current]['name']}")

        # 셀프플레이 - 학습 에이전트 외의 나머지 에이전트 업데이트
        if mean_rew > SELFPLAY_THRESH:  # 에피소드 평균 보상이 임계값을 넘으면
            print("---- Updating opponents (self-play)!!! ----")
            # 밀어내기 느낌으로 나머지 에이전트들의 정책을 업데이트
            trainer.set_weights(
                {
                    "opponent_3": trainer.get_weights(["opponent_2"])["opponent_2"],  # 더 과거
                    "opponent_2": trainer.get_weights(["opponent_1"])["opponent_1"],  # 과거
                    "opponent_1": trainer.get_weights(["default"])["default"],  # 최신
                }
            )

if __name__ == "__main__":

    ### 환경 세팅 코드 ###
    ray.init(include_dashboard=False)
    register_env("Soccer", create_rllib_env)

    temp_env = create_rllib_env({"num_envs_per_worker": NUM_ENVS_PER_WORKER})
    obs_space = temp_env.observation_space
    act_space = temp_env.action_space
    temp_env.close()
    #########################

    # 학습 알고리즘 코드
    analysis = tune.run(
        "PPO",  # 사용 알고리즘
        name="PPO_selfplay_twos",  # 결과가 저장될 폴더명
        config={
            "num_gpus": 1,  # 학습에 GPU 1개 사용
            "num_workers": 0,  # 별도 워커 없이 메인 프로세스가 데이터 수집도 겸함 (로컬 학습용)
            "num_envs_per_worker": NUM_ENVS_PER_WORKER,  # 한 번에 병렬로 돌릴 환경(경기) 수
            "log_level": "INFO",  # 진행 상황을 터미널에 출력
            "framework": "torch",  # 딥러닝 프레임워크로 PyTorch 사용
            "callbacks": CurriculumSelfPlayUpdateCallback,
            "multiagent": {  # 멀티 에이전트 설정
                "policies": {  # 이 환경에서 사용할 모든 정책 목록
                    "default":    (None, obs_space, act_space, {}),  # 학습 모델
                    "opponent_1": (None, obs_space, act_space, {}),  # 과거 자신1
                    "opponent_2": (None, obs_space, act_space, {}),  # 과거 자신2(더 과거)
                    "opponent_3": (None, obs_space, act_space, {}),  # 과거 자신3(더 더 과거)
                },
                "policy_mapping_fn": tune.function(policy_mapping_fn),  # default가 누구랑 경기할 지 결정
                "policies_to_train": ["default"],  # default 정책만 학습시키겠다는 의미
            },

            # 환경 설정
            "env": "Soccer",
            "env_config": {
                "num_envs_per_worker": NUM_ENVS_PER_WORKER,
            },

            # 신경망 모델 설정
            "model": {
                "vf_share_layers": True,  # Actor(행동)와 Critic(가치) 네트워크가 앞의 레이어를 공유
                "fcnet_hiddens": [256, 256],  # 은닉층 구조
                "fcnet_activation": "relu",  # 활성화 함수
            },

            # PPO 하이퍼파라미터
            "rollout_fragment_length": 1000,  # 한 번에 수집할 데이터 조각 길이
            "train_batch_size": 20000,  # 한 iteration 마다 학습을 위한 총 데이터 개수
            "sgd_minibatch_size": 4000,  # GPU 업데이트 시 한 번에 넣을 데이터 개수
            "num_sgd_iter": 10,  # 모은 데이터를 10번 재사용해서 학습
            "lr": 1e-4,  # 학습률
            "entropy_coeff": 0.01,  # 탐험 계수 (다양한 시도를 하기 위함)
        },
        stop={
            "timesteps_total": 10000000,  # 총 1000만 스텝을 훈련 시 종료
        },

        # 체크포인트 파일 저장 설정
        checkpoint_freq=10,
        checkpoint_at_end=True,
        local_dir="./ray_results(PPO)",
    )

    # 전체 실험 중 평균 보상(점수)이 가장 높았던 시점의 정보를 가져옴
    best_trial = analysis.get_best_trial("episode_reward_mean", mode="max")
    print(best_trial)

    # 가장 성적이 좋았던 체크포인트(모델 파일) 경로를 가져옴
    best_checkpoint = analysis.get_best_checkpoint(
        trial=best_trial, metric="episode_reward_mean", mode="max"
    )
    print(best_checkpoint)
    print("Done training")
