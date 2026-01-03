import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import numba as nb
from scipy.stats import norm

def BS_call(S0, K, T, r, sig):
    """
    Black scholes price for a European call option
    
    :param S0: Initial stock price
    :param K: Strike price
    :param T: Time of expiration
    :param r: Interest rate
    :param sig: Volatility
    """
    d1 = (np.log(S0 / K) + (r + 0.5 * sig**2) * T) / (sig * np.sqrt(T))
    d2 = d1 - sig * np.sqrt(T)

    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)



