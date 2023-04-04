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
import matplotlib.ticker as mtick

from src.products import Vanilla, Future, Index, Parameter
from src.utils import dataset_split
from src.arquitectures import AutoencoderVanillaFutureParameter, FNNVanillaFutureParameter, FNNParameter, RNNVanillaFutureParameter
from src.metrics import MaxAbsoluteError, MaxAbsolutePercentageError

import copy

# Load dataset
project_folder = "/mnt/netapp2/Store_uni/home/ulc/id/amh/calibration/calibration/" 
dataset_folder = project_folder+"datasets/dataset_commodity5/"

dates = os.listdir(dataset_folder) 

vanilla_volatilities = []
index_volatilities = []
futures = []
parameters = []
for i in range(len(dates)):
    numbers = [ int(f.split("_")[1].split(".")[0]) for f in os.listdir(dataset_folder+dates[i]+"/") if (f[0:10]=="parameters") ]
    p_dates = dataset_folder+dates[i]+"/"+"index/index_dates.csv"
    p_moneyness = dataset_folder+dates[i]+"/"+"index/moneyness.csv"
    for j in range(len(numbers)):
        p_volatilities = dataset_folder+dates[i]+"/"+"index/volatilities_"+str(numbers[j])+".csv"
        p_parameters = dataset_folder+dates[i]+"/parameters_"+str(numbers[j])+".csv"

        vanilla = Vanilla.from_folder(dates[i],dataset_folder+dates[i]+"/")
        future = Future.from_folder(dates[i],dataset_folder+dates[i]+"/")
        index = Index(dates[i],p_dates,p_moneyness,p_volatilities)
        parameter = Parameter(p_parameters)

        if ((index.volatilities<1e-5).any() or (index.volatilities>10).any()):
            print("Index ",i," Element: ",dates[i])
        else:
            vanilla_volatilities.append(index.volatilities-vanilla.volatilities)
            futures.append(future.prices)
            index_volatilities.append(index.volatilities)
            parameters.append(parameter.parameters[[0,1,4,5]])

# Convert to array
vanilla_volatilities = np.array(vanilla_volatilities)
futures = np.array(futures)
index_volatilities = np.array(index_volatilities)
parameters = np.array(parameters)

# Shapes
k_index = index.moneyness
k_vanilla = vanilla.moneyness
shape_futures = future.prices.shape
shape_parameters = (4,) 

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
shape = [120,114,108,102,98]

print("Shape vanilla: ",np.shape(k_vanilla))
print("Shape futures: ",shape_futures)
print("Shape index: ",np.shape(k_index))

model = FNNVanillaFutureParameter(k_vanilla,shape_futures,shape_parameters,k_index,shape,activation = "elu")

lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay([200,400,2000],[0.001,0.0005,0.0001,0.00005])
max_ae = MaxAbsoluteError
max_ape = MaxAbsolutePercentageError
mse = tf.keras.losses.MeanSquaredError()
mae = tf.keras.losses.MeanAbsoluteError()
mape = tf.keras.losses.MeanAbsolutePercentageError()
optimizer = tf.keras.optimizers.Adam(learning_rate = lr)
model.compile(optimizer = optimizer,
              loss=mse,
              metrics=[mae,mape,max_ape])


epochs = 4000
history = model.fit((vanilla_volatilities_train,futures_train,parameters_train),index_volatilities_train, 
        epochs=epochs, batch_size = 2**6, 
        validation_data = ((vanilla_volatilities_test,futures_test,parameters_test),index_volatilities_test))
model.summary()

plt.plot(np.arange(1,epochs+1),history.history["mean_absolute_error"],label = "loss_mae")
plt.plot(np.arange(1,epochs+1),history.history["val_mean_absolute_error"],label = "val_mae")
plt.grid()
plt.legend()
plt.yscale("log")
plt.show()
print(history.history.keys())

#np.savetxt("ind_fut_par_training.dat",
#    np.concatenate(
#        (np.arange(1,epochs+1)[:,None],
#        np.array(history.history["mean_absolute_error"])[:,None],
#        np.array(history.history["val_mean_absolute_error"])[:,None]), 
#        axis = 1),
#    delimiter = " ",header = "epochs train test",comments = "")

# Plot surfaces
maturities = (index.maturities*365/30).astype(int)
volatilities = index.volatilities
moneyness = index.moneyness[0]

index_volatilities_pred = model((vanilla_volatilities_test,futures_test,parameters_test))
error_mape = tf.math.reduce_mean(100*tf.math.abs((index_volatilities_pred-index_volatilities_test)/index_volatilities_test),axis = 0)
error_mape_std = tf.math.reduce_std(100*tf.math.abs((index_volatilities_pred-index_volatilities_test)/index_volatilities_test),axis = 0)
error_max = tf.math.reduce_max(100*tf.math.abs((index_volatilities_pred-index_volatilities_test)/index_volatilities_test),axis = 0)


fig = plt.figure()

ax1=fig.add_subplot(1,3,1)
ax1.set_title("Average relative error",fontsize=15,y=1.04)
im1 = ax1.imshow(error_mape)
fig.colorbar(im1)

ax1.set_xticks(np.linspace(0,len(moneyness)-1,len(moneyness)))
ax1.set_xticklabels(moneyness)
ax1.set_yticks(np.linspace(0,len(maturities)-1,len(maturities)))
ax1.set_yticklabels(maturities)
ax1.set_xlabel("Strike",fontsize=15,labelpad=5)
ax1.set_ylabel("Maturity",fontsize=15,labelpad=5)

ax2=fig.add_subplot(1,3,2)
ax2.set_title("Std relative error",fontsize=15,y=1.04)
im2 = ax2.imshow(error_mape_std)
fig.colorbar(im2)

ax2.set_xticks(np.linspace(0,len(moneyness)-1,len(moneyness)))
ax2.set_xticklabels(moneyness)
ax2.set_yticks(np.linspace(0,len(maturities)-1,len(maturities)))
ax2.set_yticklabels(maturities)
ax2.set_xlabel("Strike",fontsize=15,labelpad=5)
ax2.set_ylabel("Maturity",fontsize=15,labelpad=5)

ax3=fig.add_subplot(1,3,3)
ax3.set_title("Max error",fontsize=15,y=1.04)
im3 = ax3.imshow(error_max)
fig.colorbar(im3)

ax3.set_xticks(np.linspace(0,len(moneyness)-1,len(moneyness)))
ax3.set_xticklabels(moneyness)
ax3.set_yticks(np.linspace(0,len(maturities)-1,len(maturities)))
ax3.set_yticklabels(maturities)
ax3.set_xlabel("Strike",fontsize=15,labelpad=5)
ax3.set_ylabel("Maturity",fontsize=15,labelpad=5)

plt.show()



## Choose what to plot

#model.plot(["worst","best","median"],(vanilla_volatilities_test,futures_test,parameters_test),index_volatilities_test,save_data = False,make_plot = True)
