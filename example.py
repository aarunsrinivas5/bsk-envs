import numpy as np
from bsk_transfers import HohmannTransferEnv

env = HohmannTransferEnv(
    fidelity='low',
    max_steps=100, 
    max_delta_v=10000, 
    render_mode='human'
)

state, _ = env.reset()
for i in range(200):
    state, reward, done, _, info = env.step(0)