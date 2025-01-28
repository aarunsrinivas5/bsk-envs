import gymnasium as gym
import numpy as np
from gymnasium import spaces
from interplanetary_transfer.low.env import InterplanetaryTransfer1DOFEnv
from interplanetary_transfer.medium.env import InterplanetaryTransfer3DOFEnv
from interplanetary_transfer.high.env import InterplanetaryTransfer6DOFEnv

class InterplanetaryTransferEnv(gym.Env):

    def __init__(self, fidelity='low', max_steps=100, max_delta_v=10000, render_mode=None):
        super(InterplanetaryTransferEnv, self).__init__()
        if fidelity == 'low':
            self.env = InterplanetaryTransfer1DOFEnv(
                max_steps=max_steps, 
                max_delta_v=max_delta_v, 
                render_mode=render_mode
            )
        if fidelity == 'medium':
            self.env = InterplanetaryTransfer3DOFEnv(
                max_steps=max_steps, 
                max_delta_v=max_delta_v, 
                render_mode=render_mode
            )
        if fidelity == 'high':
            self.env = InterplanetaryTransfer6DOFEnv(
                max_steps=max_steps, 
                max_delta_v=max_delta_v, 
                render_mode=render_mode
            )
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def reset(self, seed=None, options={}):
        return self.env.reset(seed, options)
    
    def step(self, action):
        return self.env.step(action)