from gymnasium.envs.registration import register

register(
    id='OrbitDiscovery3DOF-v0', 
    entry_point='bsk_envs.orbit_discovery.medium.env:OrbitDiscovery3DOFEnv'
)