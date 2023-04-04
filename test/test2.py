################################################
# Test autoencoder: vanillas+futures ----> index
################################################
import sys
sys.path.append("./")
sys.path.append("../")
import os

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from src.products import Vanilla, Future, Index
from src.utils import dataset_split
from src.arquitectures import AutoencoderVanillaFuture

import copy

# Load dataset
project_folder = "/mnt/netapp2/Store_uni/home/ulc/id/amh/calibration/calibration/" 
dataset_folder = project_folder+"datasets/dataset_commodity1/"

dates = os.listdir(dataset_folder) 
shape_vanillas = Vanilla.from_folder(dates[0],dataset_folder+dates[0]+"/").volatilities.shape
shape_futures = Future.from_folder(dates[0],dataset_folder+dates[0]+"/").prices.shape
shape_index = Index.from_folder(dates[0],dataset_folder+dates[0]+"/").volatilities.shape

vanilla_volatilities = []
index_volatilities = []
futures = []
for i in range(len(dates)):
    vanilla = Vanilla.from_folder(dates[i],dataset_folder+dates[i]+"/")
    future = Future.from_folder(dates[i],dataset_folder+dates[i]+"/")
    index = Index.from_folder(dates[i],dataset_folder+dates[i]+"/")

    if ((index.volatilities<0).any() or (index.volatilities>5).any()):
        print("Index ",i," Element ",dates[i],": ",np.max(np.abs(index.volatilities)))
    else:
        vanilla_volatilities.append(vanilla.volatilities)
        futures.append(future.prices)
        index_volatilities.append(index.volatilities)

vanilla_volatilities = np.array(vanilla_volatilities)
futures = np.array(futures)
index_volatilities = np.array(index_volatilities)

# Train and test split
train, test = dataset_split((vanilla_volatilities,futures,index_volatilities))

vanilla_volatilities_train = train[0]
futures_train = train[1]
index_volatilities_train = train[2]

vanilla_volatilities_test = test[0]
futures_test = test[1]
index_volatilities_test = test[2]


# Neural network definition
shape_encoder_vanillas = [64,32,16,8]
shape_encoder_futures = [10,5,3]
shape_decoder = [11,16,32,64]

model = AutoencoderVanillaFuture(shape_vanillas,shape_futures,shape_index,shape_encoder_vanillas,shape_encoder_futures,shape_decoder)

optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)
model.compile(optimizer = optimizer,
              loss='mean_squared_error',
              metrics=['mean_absolute_error'])

# Fit
epochs = 2000
history = model.fit((vanilla_volatilities_train,futures_train),index_volatilities_train, 
        epochs=epochs, batch_size = 2**16, 
        validation_data = ((vanilla_volatilities_test,futures_test),index_volatilities_test))
np.savetxt("ind_fut_training.dat",
    np.concatenate(
        (np.arange(1,epochs+1)[:,None],
        np.array(history.history["mean_absolute_error"])[:,None],
        np.array(history.history["val_mean_absolute_error"])[:,None]), 
        axis = 1),
    delimiter = " ",header = "epochs train test",comments = "")
sys.exit(0)

## Choose what to plot
index_plot = 0
y_pred = model.predict((vanilla_volatilities_test[index_plot:index_plot+1],futures_test[index_plot:index_plot+1]))[0]
y_target = index_volatilities_test[index_plot]

# Save
for i in range(y_target.shape[0]):
    np.savetxt("ind_fut_"+"{}".format(i+1)+".dat",
        np.concatenate((index.moneyness[i,:][:,None],y_target[i][:,None],y_pred[i][:,None]), axis = 1),
        delimiter = " ",header = "moneyness iv_target iv_pred",comments = "")
## Plot
#fig = plt.figure()
#ax = plt.subplot(111)
#for i in range(y_target.shape[0]):
#    plt.scatter(index.moneyness[i,:],y_target[i],label = "Target "+vanilla.dates_options[i].strftime("%Y-%m-%d"))
#    plt.plot(index.moneyness[i,:],y_pred[i],label = "Pred "+vanilla.dates_options[i].strftime("%Y-%m-%d"))
#    ax.legend()
##
#ax.set_xlabel("Moneyness")
#ax.set_ylabel("Implied vol")
#plt.legend()
#plt.show()
