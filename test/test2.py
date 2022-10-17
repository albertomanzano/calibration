######################################
# Test execution time and working
######################################

import sys
sys.path.append("./")
sys.path.append("../")

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import ndtr as N

from src.model import Model
from src.drift import DriftLinear
from src.diffusion import DiffusionConstant

def bs_call(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * N(d1) - K * np.exp(-r*T)* N(d2)

# Definition of the market
T_len = 8
K_len = 10
T = tf.linspace(0.1,2.5,T_len)
K = tf.linspace(0.5,1.5,K_len)
S0 = 1.

# Black-Scholes model
r = 0.03
sigma = 0.5
bs_prices = np.zeros((K_len,T_len))
for i in range(K_len):
    for j in range(T_len):
        bs_prices[i,j] = bs_call(S0,K[i],T[j],r,sigma)

# Model 1 
drift = DriftLinear(r,0.0)
diffusion = DiffusionConstant(sigma)
model = Model(drift,diffusion)
model_prices = model.plain_vanilla(S0,T,K,r)

# Comparison
difference = np.abs(model_prices.numpy()-bs_prices)
print("Difference: ", difference)
