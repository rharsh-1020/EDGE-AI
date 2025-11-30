import numpy as np

def random_policy(env):
    return env.action_space.sample()

def greedy_policy(env):
    # Choose cluster with minimum average utilization Z
    state = env._get_state()
    h = env.h
    Z = state[:h]
    return int(np.argmin(Z))
