import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from bsk_envs.orbit_discovery.medium.sim  import OrbitDiscovery3DOFSimulator


class OrbitDiscovery3DOFEnv(gym.Env):

    metadata = {'render_modes': ['human']}

    def __init__(
            self, 
            mu=4.463e5, 
            radius=8000,
            max_steps=100,
            render_mode=None
        ):
        super(OrbitDiscovery3DOFEnv, self).__init__()
        self.mu = mu
        self.radius = radius
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.v_norm = np.sqrt(2 * mu / radius)
        self.total_delta_v = 0
        self.step_count = 0
        self.obs = None
        self.simulator = None

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,))
        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,))

    def _get_state(self):
        r = self.obs['r_S_N'] / self.radius
        v = self.obs['v_S_N'] / self.v_norm
        return np.concatenate((r, v), dtype=np.float32)

    def _get_reward(self):
        r = np.linalg.norm(self.obs['r_S_N'] / self.radius)
        if 1 < r < 10:
            return 1 - r
        return -self.max_steps
    
    def _get_terminal(self):
        if self.step_count > self.max_steps:
            return True
        r = np.linalg.norm(self.obs['r_S_N'] / self.radius)
        if 1 < r < 10:
            return False
        return True
        
    def reset(self, seed=None, options={}):
        if self.simulator is not None:
            del self.simulator
        self.simulator = OrbitDiscovery3DOFSimulator(
            mu=self.mu, 
            radius=self.radius, 
            render_mode=self.render_mode
        )
        self.obs = self.simulator.init()
        self.step_count = 0
        self.total_delta_v = 0
        state = self._get_state()
        return state, self.obs
    
    def step(self, action):
        self.obs = self.simulator.run(action / 10)
        self.total_delta_v += np.linalg.norm(action / 10)
        self.step_count += 1
        next_state = self._get_state()
        reward = self._get_reward()
        done = self._get_terminal()
        return next_state, reward, done, False, self.obs

    def render(self):
        if self.render_mode == 'human':
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111, projection='3d')

            u = np.linspace(0, 2 * np.pi, 100)
            v = np.linspace(0, np.pi, 100)
            x = self.radius * np.outer(np.cos(u), np.sin(v))
            y = self.radius * np.outer(np.sin(u), np.sin(v))
            z = self.radius * np.outer(np.ones(np.size(u)), np.cos(v))
            ax.plot_surface(x, y, z, color='b', alpha=0.3)

            ax.set_xlabel('x (m)')
            ax.set_ylabel('y (m)')
            ax.set_zlabel('z (m)')
            r_N = self.simulator.dataLog.r_BN_N
            ax.plot(r_N[:, 0], r_N[:, 1], r_N[:, 2])
            plt.pause(0.1)

    def close(self):
        if self.simulator is not None:
            del self.simulator
        if self.render_mode == 'human':
            plt.close()



    