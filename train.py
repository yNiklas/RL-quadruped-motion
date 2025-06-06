from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from logger_callback import LoggerCallback
from quadruped_env import QuadrupedEnv
import torch

print("Using device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

env = make_vec_env(QuadrupedEnv, n_envs=16)
log_cb = LoggerCallback()

#model = PPO("MlpPolicy", env, device="cuda", verbose=1, tensorboard_log="./quadruped_tensorboard/", clip_range=0.1)
model = PPO.load("quadruped_ppo_model", env=env, device="cuda", verbose=1, tensorboard_log="./quadruped_tensorboard/")
model.learn(total_timesteps=1_000_000, callback=log_cb)
model.save("quadruped_ppo_model")
