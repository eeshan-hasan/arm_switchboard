import numpy as np

def identity(x):
    return x

def sigmoid(x):
    x = np.asarray(x)
    return np.where(
        x >= 0,
        1 / (1 + np.exp(-x)),
        np.exp(x) / (1 + np.exp(x))
    )
def power_10(x):
    return 10**x

def log_10(x):
    return np.log10(x)

def logit(x):
    return np.log(x/(1-x))