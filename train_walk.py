from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from logger_callback import LoggerCallback
import torch

from walker_quadruped_env import WalkerQuadrupedEnv

print("Using device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

env = make_vec_env(WalkerQuadrupedEnv, n_envs=32)
log_cb = LoggerCallback()

#model = PPO("MlpPolicy", env, device="cuda", verbose=1, tensorboard_log="./quadruped_tensorboard/", clip_range=0.1)
model = PPO.load("quadruped_ppo_model_gauss", env=env, device="cpu", verbose=1, tensorboard_log="./quadruped_tensorboard/")
model.learn(total_timesteps=1_000_000, callback=log_cb)
model.save("quadruped_ppo_model_gauss")
