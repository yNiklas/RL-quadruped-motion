from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from quadruped_env import QuadrupedEnv
import torch

print("Using device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

env = make_vec_env(QuadrupedEnv, n_envs=8)

model = PPO("MlpPolicy", env, device="cuda", verbose=1, tensorboard_log="./quadruped_tensorboard/")
model.learn(total_timesteps=1_000_000)
model.save("quadruped_ppo_model")
