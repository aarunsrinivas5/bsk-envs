import numpy as np
import gymnasium as gym
import bsk_envs
from stable_baselines3 import PPO


env = gym.make('OrbitDiscovery3DOF-v0', render_mode='human')
model = PPO.load(f'baselines/orbit_discovery-medium', env)

done = False
state, _ = env.reset()
total_reward = 0
while not done:
    action, _ = model.predict(state, deterministic=True)
    state, reward, done, _, info = env.step(action)
    total_reward += reward
    env.render()
env.close()



