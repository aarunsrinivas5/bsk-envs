from gymnasium.envs.registration import register

register(
    id='InterplanetaryTransfer6DOF-v0', 
    entry_point='bsk_envs.interplanetary_transfer.high.env:InterplanetaryTransfer6DOFEnv'
)