import gymnasium as gym
import numpy as np
from gymnasium import spaces
from bsk_envs.interplanetary_transfer.medium.sim import InterplanetaryTransfer3DOFSimulator


THRESHOLD = 50e6 * 1000
RE = 149.78e6 * 1000
RM = 228e6 * 1000
RMIN = RE - THRESHOLD
RMAX = RM + THRESHOLD

class InterplanetaryTransfer3DOFEnv(gym.Env):

    metadata = {'render_modes': ['human']}

    def __init__(self, max_steps=100, max_delta_v=10000, render_mode=None):
        super(InterplanetaryTransfer3DOFEnv, self).__init__()
        self.max_steps = max_steps
        self.max_delta_v = max_delta_v
        self.render_mode = render_mode
        self.total_delta_v = 0
        self.step_count = 0
        self.obs = None
        self.simulator = None

        self.observation_space = spaces.Box(low=-1e16, high=1e16, shape=(7,))
        self.action_space = spaces.Box(-1, 1, shape=(3,))

    def _get_state(self):
        r = (self.obs['r_S_N'] - self.obs['r_M_N']) / np.linalg.norm(self.obs['r_M_N'])
        v = (self.obs['v_S_N'] - self.obs['v_M_N']) / np.linalg.norm(self.obs['v_M_N'])
        dv = [self.total_delta_v / self.max_delta_v]
        return np.concatenate((r, v, dv))
    
    def _get_reward(self, action):
        distance_penalty = np.linalg.norm(self.obs['r_S_N'] - self.obs['r_M_N']) / np.linalg.norm(self.obs['r_M_N'])
        velocity_penalty = np.linalg.norm(self.obs['v_S_N'] - self.obs['v_M_N']) / np.linalg.norm(self.obs['v_M_N'])
        terminal_penalty = max(0, max(distance_penalty, velocity_penalty) - 1e-2)
        delta_v_penalty = max(0, abs(action) - 1e-3)
        if terminal_penalty == 0:
            return 100
        if self.step_count == self.max_steps:
            return - 100 * terminal_penalty
        if np.linalg.norm(self.obs['r_S_N'])  < RMIN:
            return -1000
        if np.linalg.norm(self.obs['r_S_N']) > RMAX:
            return -1000
        if self.total_delta_v > self.max_delta_v:
            return -1000
        return - 10 * delta_v_penalty 
    
    def _get_terminal(self):
        if self.step_count == self.max_steps:
            return True
        if np.linalg.norm(self.obs['r_S_N'])  < RMIN:
            return True
        if np.linalg.norm(self.obs['r_S_N']) > RMAX:
            return True
        if self.total_delta_v > self.max_delta_v:
            return True
        return False
        
    def reset(self, seed=None, options={}):
        if self.simulator is not None:
            del self.simulator
        self.simulator = InterplanetaryTransfer3DOFSimulator(
            render_mode=self.render_mode
        )
        self.obs = self.simulator.init()
        self.step_count = 0
        state = self._get_state()
        return state, {}
    
    def step(self, action):
        self.obs = self.simulator.run(action * self.max_delta_v)
        self.total_delta_v += abs(np.linalg.norm(action * self.max_delta_v))
        self.step_count += 1
        next_state = self._get_state()
        reward = self._get_reward(action)
        done = self._get_terminal()
        return next_state, reward, done, False, {}
    
    def close(self):
        if self.simulator is not None:
            del self.simulator
    
    
