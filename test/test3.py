######################################
# Training against synthetic data 
######################################

import sys
sys.path.append("./")
sys.path.append("../")

import tensorflow as tf
import numpy as np
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
discounting = 0.0


# Model structure 
a = 1.2
b = 0.0
neurons_layer = 3
layers = 3

# Model 1
drift1 = DriftLinear(a,b)
diffusion1 = DiffusionFCNN(layers,neurons_layer)
model1 = Model(drift1,diffusion1)
reference_prices = model1.plain_vanilla(S0,T,K, discounting)

# Model 2
drift2 = DriftLinear(a,b)
diffusion2 = DiffusionFCNN(layers,neurons_layer)
model2 = Model(drift2,diffusion2)

# Fit
epochs = 100
opt = tf.keras.optimizers.Adam(learning_rate=0.01)
loss = lambda: model2.plain_vanilla_loss(S0,T,K,discounting,reference_prices)
for i in range(epochs):
    opt.minimize(loss, [model2.diffusion.trainable_parameters])
    error = model2.plain_vanilla_loss(S0,T,K,discounting,reference_prices)
    print("Iteration ",i," : ",error)

fitted_prices = model2.plain_vanilla(S0,T,K, discounting)

# Comparison
print("Fitted prices: ", np.abs(fitted_prices.numpy()-reference_prices.numpy()))
for i in range(T_len):
    plt.plot(K,fitted_prices[:,i],label = "Fit "+str(T[i].numpy()))
    plt.scatter(K,reference_prices[:,i],label = "Ref "+str(T[i].numpy()))

plt.xlabel("K")
plt.xlabel("V")
plt.grid()
plt.legend(bbox_to_anchor=(1.01, 1.0), loc='upper left')
plt.show()


