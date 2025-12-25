from stable_baselines3 import PPO, A2C

m_blue = PPO.load("PPO/models_dual_final_blue/dualppo_latest.zip", device="cpu")
print("BLUE total steps(PPO):", m_blue.num_timesteps)

m_orange = PPO.load("PPO/models_dual_final_orange/dualppo_latest.zip", device="cpu")
print("ORANGE total steps(PPO):", m_orange.num_timesteps)

m_blue = A2C.load("A2C/models_dual_final_blue/duala2c_latest.zip", device="cpu")
print("BLUE total steps(A2C):", m_blue.num_timesteps)

# m_orange = A2C.load("A2C/models_dual_final_orange/duala2c_latest.zip", device="cpu")
# print("ORANGE total steps(A2C):", m_orange.num_timesteps)


