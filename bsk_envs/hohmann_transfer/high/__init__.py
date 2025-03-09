from gymnasium.envs.registration import register

register(
    id='HohmannTransfer6DOF-v0', 
    entry_point='bsk_envs.hohmann_transfer.high.env:HohmannTransfer6DOFEnv'
)