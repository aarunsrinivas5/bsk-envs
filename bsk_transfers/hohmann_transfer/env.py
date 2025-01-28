import gymnasium as gym
import numpy as np
from gymnasium import spaces
from .low.env import HohmannTransfer1DOFEnv
from .medium.env import HohmannTransfer3DOFEnv
from .high.env import HohmannTransfer6DOFEnv

class HohmannTransferEnv(gym.Env):

    def __init__(self, fidelity='low', max_steps=100, max_delta_v=10000, render_mode=None):
        super(HohmannTransferEnv, self).__init__()
        if fidelity == 'low':
            self.env = HohmannTransfer1DOFEnv(
                max_steps=max_steps, 
                max_delta_v=max_delta_v, 
                render_mode=render_mode
            )
        if fidelity == 'medium':
            self.env = HohmannTransfer3DOFEnv(
                max_steps=max_steps, 
                max_delta_v=max_delta_v, 
                render_mode=render_mode
            )
        if fidelity == 'high':
            self.env = HohmannTransfer6DOFEnv(
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