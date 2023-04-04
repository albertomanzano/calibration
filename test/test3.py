################################################
# Test autoencoder: vanillas+futures+parameters ----> index
################################################
import sys
sys.path.append("./")
sys.path.append("../")
import os

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from src.products import Vanilla, Future, Index, Parameter
from src.utils import dataset_split
from src.arquitectures import AutoencoderVanillaFutureParameter
from src.losses import MaxLoss

import copy

# Load dataset
project_folder = "/mnt/netapp2/Store_uni/home/ulc/id/amh/calibration/calibration/" 
dataset_folder = project_folder+"datasets/dataset_commodity4/"

dates = os.listdir(dataset_folder) 
k_vanilla = Vanilla.from_folder(dates[0],dataset_folder+dates[0]+"/").moneyness
shape_futures = Future.from_folder(dates[0],dataset_folder+dates[0]+"/").prices.shape
k_index = Index.from_folder(dates[0],dataset_folder+dates[0]+"/").moneyness
shape_parameters = (4,) 

vanilla_volatilities = []
index_volatilities = []
futures = []
parameters = []
for i in range(len(dates)):
    vanilla = Vanilla.from_folder(dates[i],dataset_folder+dates[i]+"/")
    future = Future.from_folder(dates[i],dataset_folder+dates[i]+"/")
    index = Index.from_folder(dates[i],dataset_folder+dates[i]+"/")
    parameter = Parameter.from_folder(dataset_folder+dates[i]+"/")

    if ((index.volatilities<1e-5).any() or (index.volatilities>5).any()):
        print("Index ",i," Element: ",dates[i])
    else:
        vanilla_volatilities.append(vanilla.volatilities)
        futures.append(future.prices)
        index_volatilities.append(index.volatilities)
        parameters.append(parameter.parameters[[0,1,4,5]])

vanilla_volatilities = np.array(vanilla_volatilities)
futures = np.array(futures)
futures = (futures-np.mean(futures,axis = 0))/np.std(futures,axis = 0)
index_volatilities = np.array(index_volatilities)
parameters = np.array(parameters)

# Train and test split
train, test = dataset_split((vanilla_volatilities,futures,index_volatilities,parameters))


vanilla_volatilities_train = train[0]
futures_train = train[1]
index_volatilities_train = train[2]
parameters_train = train[3]

vanilla_volatilities_test = test[0]
futures_test = test[1]
index_volatilities_test = test[2]
parameters_test = test[3]


print("Len train: ",len(parameters_train))
print("Len test: ",len(parameters_test))

# Neural network definition
shape_encoder_vanillas = [64,32,16,8]
shape_encoder_futures = [10,7,4]
shape_decoder = [16,32,64]

model = AutoencoderVanillaFutureParameter(k_vanilla,shape_futures,shape_parameters,k_index,shape_encoder_vanillas,shape_encoder_futures,shape_decoder)

max_loss = MaxLoss()
optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)
model.compile(optimizer = optimizer,
              loss=max_loss,
              metrics=['mean_absolute_error','mean_squared_error'])


epochs = 2000
history = model.fit((vanilla_volatilities_train,futures_train,parameters_train),index_volatilities_train, 
        epochs=epochs, batch_size = 2**16, 
        validation_data = ((vanilla_volatilities_test,futures_test,parameters_test),index_volatilities_test))

np.savetxt("ind_fut_par_training.dat",
    np.concatenate(
        (np.arange(1,epochs+1)[:,None],
        np.array(history.history["mean_absolute_error"])[:,None],
        np.array(history.history["val_mean_absolute_error"])[:,None]), 
        axis = 1),
    delimiter = " ",header = "epochs train test",comments = "")

## Choose what to plot

model.plot("worst",(vanilla_volatilities_test,futures_test,parameters_test),index_volatilities_test,save_data = True)
model.plot("best",(vanilla_volatilities_test,futures_test,parameters_test),index_volatilities_test,save_data = True)
