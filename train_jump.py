from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from sb3_contrib import RecurrentPPO

from jumper_quadruped_env import JumperQuadrupedEnv
from logger_callback import LoggerCallback

env = make_vec_env(JumperQuadrupedEnv, n_envs=16)
log_cb = LoggerCallback()

#model = RecurrentPPO("MlpLstmPolicy", env, device="cuda", verbose=1, tensorboard_log="./quadruped_tensorboard/")
model = RecurrentPPO.load("quadruped_lstm_ppo_jump", env=env, device="cuda", verbose=1, tensorboard_log="./quadruped_tensorboard/")
model.learn(total_timesteps=150_000, callback=log_cb)
model.save("quadruped_lstm_ppo_jump")
