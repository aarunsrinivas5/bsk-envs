import gymnasium as gym
import numpy as np
from gymnasium import spaces
from Basilisk.architecture import astroConstants
from sim import HohmannTransfer1DOFSimulator


THRESHOLD = 1000 * 1000
R1 = 7000 * 1000
R2 = 42000 * 1000
V2 = np.sqrt(astroConstants.MU_EARTH / R2)
RMIN = R1 - THRESHOLD
RMAX = R2 + THRESHOLD * 10

class HohmannTransfer1DOFEnv(gym.Env):

    def __init__(self, max_steps=100, max_delta_v=10000, render_mode=None):
        super(HohmannTransfer1DOFEnv, self).__init__()
        self.max_steps = max_steps
        self.max_delta_v = max_delta_v
        self.render_mode = render_mode
        self.total_delta_v = 0
        self.step_count = 0
        self.obs = None
        self.simulator = None

        self.observation_space = spaces.Box(low=-1e16, high=1e16, shape=(7,))
        self.action_space = spaces.Box(-1, 1, shape=(1,))

    def _get_state(self):
        r = self.obs['r_S_N'] / R2
        v = self.obs['v_S_N'] / V2
        dv = [self.total_delta_v / self.max_delta_v]
        return np.concatenate((r, v, dv))
    
    def _get_reward(self, action):
        R = np.linalg.norm(self.obs[:3])
        distance_penalty = abs(R - R2) / R2
        delta_v_penalty = abs(action[0])
        penalty = distance_penalty + delta_v_penalty
        if self.step_count == self.max_steps:
            return distance_penalty * 10
        if R < RMIN:
            return -1000
        if R > RMAX:
            return -1000
        return -penalty
    
    def _get_terminal(self):
        R = np.linalg.norm(self.obs[:3])
        if self.step_count == self.max_steps:
            return True
        if R < RMIN:
            return True
        if R > RMAX:
            return True
        return False
        
    def reset(self, seed=None, options={}):
        if self.simulator is not None:
            del self.simulator
        self.simulator = HohmannTransfer1DOFSimulator(
            render_mode=self.render_mode
        )
        self.obs, info = self.simulator.init()
        self.step_count = 0
        state = self._get_state()
        return state, info
    
    def step(self, action):
        self.obs, info = self.simulator.run(action * self.max_delta_v)
        self.total_delta_v += abs(action * self.max_delta_v)
        self.step_count += 1
        next_state = self._get_state()
        reward = self._get_reward(action)
        done = self._get_terminal()
        return next_state, reward, done, False, info