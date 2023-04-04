################################################
# Test autoencoder: vanillas ---> vanillas
################################################
import sys
sys.path.append("./")
sys.path.append("../")
import os
from functools import partial
from multiprocessing import Pool

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from src.products import Vanilla
from src.utils import dataset_split
from src.arquitectures import Autoencoder

import copy

# Load dataset
project_folder = "/mnt/netapp2/Store_uni/home/ulc/id/amh/calibration/calibration/" 
dataset_folder = project_folder+"datasets/dataset_commodity4/"

dates = os.listdir(dataset_folder) 
vanilla = Vanilla.from_folder(dates[0],dataset_folder+dates[0]+"/")
shape = vanilla.volatilities.shape
volatilities = [] 
for i in range(len(dates)):
    vanilla = Vanilla.from_folder(dates[i],dataset_folder+dates[i]+"/")
    volatilities.append(vanilla.volatilities)

volatilities = np.array(volatilities)


(x_train, y_train), (x_test, y_test) = dataset_split((volatilities,volatilities))
print("Len train: ",len(x_train))
print("Len test: ",len(x_test))


# Neural network definition
shape_input = shape 
shape_output = shape
shape_encoder = [64,32,16,8]
shape_decoder = [8,16,32,64]
model = Autoencoder(shape_input,shape_output,shape_encoder,shape_decoder)

optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)
model.compile(optimizer = optimizer,
              loss='mean_squared_error',
              metrics=['mean_absolute_error'])


# Fit
epochs = 2000
history = model.fit(x_train, y_train, epochs=epochs, batch_size = 2**16, validation_data = (x_test,y_test))

# Save training
np.savetxt("vanilla_training.dat",
    np.concatenate(
        (np.arange(1,epochs+1)[:,None],
        np.array(history.history["mean_absolute_error"])[:,None],
        np.array(history.history["val_mean_absolute_error"])[:,None]), 
        axis = 1),
    delimiter = " ",header = "epochs train test",comments = "")

model.plot(index_plot = "worst",x = x_test,k = vanilla.moneyness,save_data = True)
model.plot(index_plot = "best",x = x_test,k = vanilla.moneyness,save_data = True)
