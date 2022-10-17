######################################
# Test execution time and working
######################################

import sys
sys.path.append("./")
sys.path.append("../")

import tensorflow as tf
import matplotlib.pyplot as plt
import time

from src.model import Model
from src.drift import DriftLinear
from src.diffusion import DiffusionFCNN
# Definition of the market
T_len = 10
K_len = 10
T = tf.linspace(0.1,2.5,T_len)
K = tf.linspace(0.5,1.5,K_len)
S0 = 1.


# Model 1 
drift = DriftLinear(1.2,0.0)
diffusion = DiffusionFCNN(3,3)

model = Model(drift,diffusion)

# Time with compilation
start = time.time()
prices_model = model.plain_vanilla(S0,T,K)
end = time.time()
print("Time with compilation: ", end-start)

# Time without compilation
start = time.time()
prices_model = model.plain_vanilla(S0,T,K)
end = time.time()
print("Time without compilation: ", end-start)
