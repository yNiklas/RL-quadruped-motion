from sb3_contrib import RecurrentPPO
from stable_baselines3.common.env_util import make_vec_env

from logger_callback import LoggerCallback

from recurrent_walker_quadruped_env import RecurrentWalkerEnv

env = make_vec_env(RecurrentWalkerEnv, n_envs=16)
log_cb = LoggerCallback()

#model = RecurrentPPO("MlpLstmPolicy", env, device="cuda", verbose=1, tensorboard_log="./quadruped_tensorboard/", clip_range=0.1)
model = RecurrentPPO.load("quadruped_lstm_ppo", env=env, device="cuda", verbose=1, tensorboard_log="./quadruped_tensorboard/")
model.learn(total_timesteps=150_000, callback=log_cb)
model.save("quadruped_lstm_ppo")
