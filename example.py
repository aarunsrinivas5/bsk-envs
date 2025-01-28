from bsk_transfers import HohmannTransferEnv

env = HohmannTransferEnv()
state, _ = env.reset()
for i in range(200):
    state, reward, done, _, info = env.step(0)
print('completed')