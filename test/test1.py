################################################
# Test autoencoder in the plain vanillas surface
################################################
import sys
sys.path.append("./")
sys.path.append("../")
import os

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from src.products import Vanilla
from src.utils import dataset_split
from src.arquitectures import Autoencoder

# Load dataset
project_folder = "/mnt/netapp2/Store_uni/home/ulc/id/amh/calibration/calibration/" 
dataset_folder = project_folder+"datasets/dataset_commodity/"

dates = os.listdir(dataset_folder) 
shape = Vanilla.from_folder(dates[0],dataset_folder+dates[0]+"/").volatilities.shape
volatilities = np.zeros((len(dates),shape[0],shape[1]))
for i in range(len(dates)):
    vanilla = Vanilla.from_folder(dates[i],dataset_folder+dates[i]+"/")
    volatilities[i] = vanilla.volatilities

x_train, y_train, x_test, y_test = dataset_split(volatilities,volatilities)


# Neural network definition
model = Autoencoder(shape,shape,[128,96,65,32,16,8])

optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)
model.compile(optimizer = optimizer,
              loss='mean_squared_error',
              metrics=['mean_absolute_error'])


# Fit
epochs = 2000
history = model.fit(x_train, y_train, epochs=epochs, batch_size = 2**16, validation_data = (x_test,y_test))

# Choose what to plot
index = 0
y_pred = model.predict(x_test[index:index+1])[0]
y_target = y_test[index]

# Plot
fig = plt.figure()
ax = plt.subplot(111)
for i in range(y_target.shape[0]):
    plt.scatter(vanilla.moneyness[i,:],y_target[i],label = "Target "+vanilla.dates_options[i].strftime("%Y-%m-%d"))
    plt.plot(vanilla.moneyness[i,:],y_pred[i],label = "Pred "+vanilla.dates_options[i].strftime("%Y-%m-%d"))
    ax.legend()

    ax.set_xlabel("Moneyness")
    ax.set_ylabel("Implied vol")
    plt.legend()
    plt.show()
