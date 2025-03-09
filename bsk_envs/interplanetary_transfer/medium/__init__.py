from gymnasium.envs.registration import register

register(
    id='InterplanetaryTransfer3DOF-v0', 
    entry_point='bsk_envs.interplanetary_transfer.medium.env:InterplanetaryTransfer3DOFEnv'
)