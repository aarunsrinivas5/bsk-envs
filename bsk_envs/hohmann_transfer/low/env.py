import gymnasium as gym
import numpy as np
from gymnasium import spaces
from Basilisk.architecture import astroConstants
from bsk_envs.hohmann_transfer.low.sim  import HohmannTransfer1DOFSimulator


THRESHOLD = 1000 * 1000
R1 = 7000 * 1000
R2 = 42000 * 1000
V1 = np.sqrt(astroConstants.MU_EARTH * 1e9 / R1)
V2 = np.sqrt(astroConstants.MU_EARTH * 1e9 / R2)
RMIN = R1 - THRESHOLD
RMAX = R2 + THRESHOLD * 20

class HohmannTransfer1DOFEnv(gym.Env):

    metadata = {'render_modes': ['human']}

    def __init__(
            self, 
            grav_body='earth', 
            max_delta_v=10000, 
            max_steps=200,
            render_mode=None
        ):
        super(HohmannTransfer1DOFEnv, self).__init__()
        self.grav_body = grav_body
        self.max_delta_v = max_delta_v
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.total_delta_v = 0
        self.num_thrusts = 0
        self.step_count = 0
        self.obs = None
        self.simulator = None

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,))
        self.action_space = spaces.Box(-1, 1, shape=(1,))

    def _cartesian_to_polar(self, r, norm):
        x, y, z = r
        r = np.linalg.norm([x, y, z]) / norm
        theta = np.arctan2(y, x) / np.pi
        phi = np.arccos(z / r) / np.pi
        return np.array([r, theta, phi])

    def _get_state(self):
        r = self._cartesian_to_polar(self.obs['r_S_N'], norm=R2)
        v = self._cartesian_to_polar(self.obs['v_S_N'], norm=V2)
        dv = [self.total_delta_v / self.max_delta_v]
        return np.concatenate((r, v, dv), dtype=np.float32)

    def _get_reward(self, action):
        distance_penalty = abs(np.linalg.norm(self.obs['r_S_N']) - R2) / R2
        velocity_penalty = abs(np.linalg.norm(self.obs['v_S_N']) - V2) / V2
        terminal_penalty = max(0, max(distance_penalty, velocity_penalty) - 1e-3)
        if not terminal_penalty or self.step_count == self.max_steps:
            return -10 * (terminal_penalty + self.step_count / self.max_steps)
        if np.linalg.norm(self.obs['r_S_N']) < RMIN:
            return -100
        if np.linalg.norm(self.obs['r_S_N']) > RMAX:
            return -100
        return -10 * abs(action)
    
    def _get_terminal(self):
        distance_penalty = abs(np.linalg.norm(self.obs['r_S_N']) - R2) / R2
        velocity_penalty = abs(np.linalg.norm(self.obs['v_S_N']) - V2) / V2
        terminal_penalty = -10 * max(0, max(distance_penalty, velocity_penalty) - 1e-3)
        if not terminal_penalty:
            return True
        if self.step_count == self.max_steps:
            return True
        if np.linalg.norm(self.obs['r_S_N']) < RMIN:
            return True
        if np.linalg.norm(self.obs['r_S_N']) > RMAX:
            return True
        return False
        
    def reset(self, seed=None, options={}):
        if self.simulator is not None:
            del self.simulator
        self.simulator = HohmannTransfer1DOFSimulator(
            grav_body=self.grav_body,
            render_mode=self.render_mode
        )
        self.obs = self.simulator.init()
        self.step_count = 0
        self.total_delta_v = 0
        state = self._get_state()
        return state, self.obs
    
    def step(self, action):
        action = action[0] if isinstance(action, np.ndarray) else action
        self.obs = self.simulator.run(action * self.max_delta_v)
        self.total_delta_v += abs(action * self.max_delta_v)
        self.step_count += 1
        next_state = self._get_state()
        reward = self._get_reward(action)
        done = self._get_terminal()
        return next_state, reward, done, False, self.obs
    
    def close(self):
        if self.simulator is not None:
            del self.simulator
    