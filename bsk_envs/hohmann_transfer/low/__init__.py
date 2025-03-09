from gymnasium.envs.registration import register

register(
    id='HohmannTransfer1DOF-v0', 
    entry_point='bsk_envs.hohmann_transfer.low.env:HohmannTransfer1DOFEnv',
    max_episode_steps=200
)