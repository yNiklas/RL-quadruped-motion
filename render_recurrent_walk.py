import time

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.env_util import make_vec_env

from recurrent_walker_quadruped_env import RecurrentWalkerEnv

env = make_vec_env(RecurrentWalkerEnv, n_envs=1)
model = RecurrentPPO.load("quadruped_lstm_ppo", env=env, device="cuda")

obs = env.reset()

for _ in range(1000):
    action, _ = model.predict(obs)
    obs, reward, done, info = env.step(action)
    env.envs[0].render()
    time.sleep(0.05)
    if done:
        print(info)
        obs = env.reset()
