import numpy as np


def future_return(rewards, gamma):
    ret = np.zeros_like(rewards)
    prev = 0
    for i, r in reversed(list(enumerate(rewards))):
        ret[i] = gamma * prev + r
        prev = ret[i]
    return ret
