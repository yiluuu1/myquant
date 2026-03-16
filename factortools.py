import numpy as np


def MAD_winsorize(x, multiplier=5):
    x_M = np.nanmedian(x)
    x_MAD = np.nanmedian(np.abs(x-x_M))
    upper = x_M + multiplier * x_MAD
    lower = x_M - multiplier * x_MAD
    x[x>upper] = upper
    x[x<lower] = lower
    return x