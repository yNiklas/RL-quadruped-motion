from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from jumper_quadruped_env import JumperQuadrupedEnv
from logger_callback import LoggerCallback

env = make_vec_env(JumperQuadrupedEnv, n_envs=16)
log_cb = LoggerCallback()

model = PPO("MlpPolicy", env, device="cpu", verbose=1, tensorboard_log="./quadruped_tensorboard/", clip_range=0.1)
#model = PPO.load("quadruped_ppo_jump", env=env, device="cuda", verbose=1, tensorboard_log="./quadruped_tensorboard/")
model.learn(total_timesteps=1_000_000, callback=log_cb)
model.save("quadruped_ppo_jump")
