from gymnasium.envs.registration import register

register(
    id='InterplanetaryTransfer1DOF-v0', 
    entry_point='bsk_envs.interplanetary_transfer.low.env:InterplanetaryTransfer1DOFEnv'
)