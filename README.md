## README

### 학습 코드
ppo.py
dqn.py
a2c.py

utils.py는 참조한 깃허브에서 내려받음

### 학습 완료 시 생성되는 폴더
ray_results(PPO)
ray_results(DQN)
ray_results(A2C)

### 데모영상 실행을 위한 폴더
ppo_agent
dqn_agent
a2c_agent
example_ppo_agent -> 깃허브에서 다운 받은 예시 폴더

각 폴더 안의 agent_ray.py에서 학습된 모델 파일 경로를 입력해야 함.
ex .\ray_results(PPO)\PPO_selfplay_twos\PPO_Soccer_6a750_00000_0_2025-11-23_09-36-14\checkpoint_000092\checkpoint-92

### 실행 방법
-> 터미널에서 실행
ex1 -  python -m soccer_twos.watch -m ppo_agent                     -> 각 팀이 동일한 모델 사용
ex2 -  python -m soccer_twos.watch -m1 ppo_agent -m2 dqn_agent      -> 각 팀이 서로 다른 모델 사용
ex3 -  python -m soccer_twos.watch -m1 ppo_agent -m2 a2c_agent

python -m soccer_twos.watch -m a2c_agent
