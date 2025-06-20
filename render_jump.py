import time

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from jumper_quadruped_env import JumperQuadrupedEnv

env = make_vec_env(JumperQuadrupedEnv, n_envs=1)
model = PPO.load("quadruped_ppo_jump", env=env, device="cuda")

obs = env.reset()

for _ in range(1000):
    action, _ = model.predict(obs)
    obs, reward, done, info = env.step(action)
    env.envs[0].render()
    time.sleep(0.05)
    if done:
        print(info)
        obs = env.reset()
