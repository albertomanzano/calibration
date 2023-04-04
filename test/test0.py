################################################
# Test autoencoder: futures ---> futures
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

from src.products import Future
from src.utils import dataset_split
from src.arquitectures import Autoencoder

import copy

# Load dataset
project_folder = "/mnt/netapp2/Store_uni/home/ulc/id/amh/calibration/calibration/" 
dataset_folder = project_folder+"datasets/dataset_commodity4/"

dates = os.listdir(dataset_folder) 
future = Future.from_folder(dates[0],dataset_folder+dates[0]+"/")
shape = future.prices.shape
futures = []
for date in dates:
    future = Future.from_folder(date,dataset_folder+date+"/")
    futures.append(future.prices)

futures = np.array(futures)
futures = (futures-np.mean(futures,axis = 0))/np.std(futures,axis = 0)
print(futures.shape)
print(np.mean(futures,axis = 0))
u,s,vh = np.linalg.svd(futures)
plt.scatter(np.arange(len(s)),s)
plt.semilogy()
plt.show()
plt.grid()
sys.exit(0)


train, test = dataset_split((futures,futures))
x_train = train[0]
y_train = train[1]
x_test = test[0]
y_test = test[1]


# Neural network definition
shape_input = shape 
shape_output = shape
value = shape_input[0]
shape_encoder = [value,value,value]
shape_decoder = [value,value,value]
model = Autoencoder(shape_input,shape_output,shape_encoder,shape_decoder)

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

# Save
# Save training
#np.savetxt("vanilla_training.dat",
#    np.concatenate(
#        (np.arange(1,epochs+1)[:,None],
#        np.array(history.history["mean_absolute_error"])[:,None],
#        np.array(history.history["val_mean_absolute_error"])[:,None]), 
#        axis = 1),
#    delimiter = " ",header = "epochs train test",comments = "")

# Save example

#for i in range(y_target.shape[0]):
#    np.savetxt("vanilla_"+"{}".format(i+1)+".dat",
#        np.concatenate((vanilla.moneyness[i,:][:,None],y_target[i][:,None],y_pred[i][:,None]), axis = 1),
#        delimiter = " ",header = "moneyness iv_target iv_pred",comments = "")


# Plot
fig = plt.figure()
ax = plt.subplot(111)

plt.scatter(np.arange(len(y_target)),y_target,label = "Target "+future.start.strftime("%Y-%m-%d"))
plt.plot(np.arange(len(y_pred)),y_pred,label = "Pred "+future.start.strftime("%Y-%m-%d"))
ax.legend()

ax.set_xlabel("Maturity")
ax.set_ylabel("Prices")
plt.legend()
plt.show()
