from gymnasium.envs.registration import register

register(
    id='HohmannTransfer3DOF-v0', 
    entry_point='bsk_envs.hohmann_transfer.medium.env:HohmannTransfer3DOFEnv'
)