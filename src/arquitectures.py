import sys
sys.path.append("./")
sys.path.append("../")

import tensorflow as tf
import copy
import numpy as np
import matplotlib.pyplot as plt

from src.utils import marginal

class NN(tf.keras.models.Model):
  def __init__(self):
      super(NN, self).__init__()

  def call(self, x):
    return self.model(x)

  def plot(self,index_plots,x,y,save_data = False,make_plot = True, metric = 'mean_absolute_error',dates=None,append='',folder = "dat/"):
    label = None 
    metric_index = np.where(np.array(self.metrics_names) == 'mean_absolute_error')[0][0]

    losses = np.zeros(y.shape[0])
    for i in range(y.shape[0]):
        tupla = tuple([ array[i][None,:] for array in x ])
        losses[i] = self.evaluate(x=tupla, y=y[i][None,:], batch_size=1,verbose = 0)[metric_index]

    if (dates is not None) and (make_plot):
        keys, values = marginal(dates[:,0],losses)
        plt.scatter(keys,values)
        plt.show()

    for index_plot in index_plots:
        if isinstance(index_plot,str):
            label = index_plot

            if index_plot == "best":
                index_plot = np.argmin(losses)
                print("Best index: ",index_plot," loss: ",losses[index_plot])
                if dates is not None:
                    print("Date: ",dates[index_plot])
            elif index_plot == "worst":
                index_plot = np.argmax(losses)
                print("Worst index: ",index_plot," loss: ",losses[index_plot])
                if dates is not None:
                    print("Date: ",dates[index_plot])
            elif index_plot == "median":
                index_plot = np.argsort(losses)[int(len(losses)/2)]
                print("Median index: ",index_plot," loss: ",losses[index_plot])
                if dates is not None:
                    print("Date: ",dates[index_plot])
            else:
                raise ValueError


        tupla = tuple([array[index_plot:index_plot+1] for array in x])
        y_pred = self.predict(tupla)[0]
        y_target = y[index_plot]


        # Save data
        if save_data:
            # Save surfaces
            if label is None: label = str(index_plot)
            header1 = ''.join(" pred_"+str(i) for i in range(y_target.shape[0]))
            header2 = ''.join(" target_"+str(i) for i in range(y_target.shape[0]))
            header = "moneyness"+header1+header2
            np.savetxt(folder+"ind_fut_par_"+label+append+".dat",
                np.concatenate((self.k_index[0,:][:,None],np.transpose(y_target),np.transpose(y_pred)), axis = 1),
                delimiter = " ",header = header,comments = "")
            #for i in range(y_target.shape[0]):
                #np.savetxt("ind_fut_par_"+"{}".format(i+1)+"_"+label+append+".dat",
                    #np.concatenate((self.k_index[i,:][:,None],y_target[i][:,None],y_pred[i][:,None]), axis = 1),
                    #delimiter = " ",header = "moneyness iv_target iv_pred",comments = "")

            # Save hist
            hist, bin_edges = np.histogram(losses, bins=10)
            np.savetxt(folder+"ind_fut_par_hist"+append+".dat",
                    np.concatenate((hist[:,None],bin_edges[:-1,None]), axis = 1),
                delimiter = " ",header = "hist bin_edges",comments = "")
        
        # Plot
        if make_plot:
            fig = plt.figure()
            ax = plt.subplot(111)
            for i in range(y_target.shape[0]):
                plt.scatter(self.k_index[i,:],y_target[i],label = "Target "+str(i+1))
                plt.plot(self.k_index[i,:],y_pred[i],label = "Pred "+str(i+1))
                ax.legend()
            #
            ax.set_xlabel("Moneyness")
            ax.set_ylabel("Implied volatility")
            plt.legend()
            plt.show()

            plt.show()



class AutoencoderVanillaFutureParameter(NN):
  def __init__(self,k_vanilla,shape_future,shape_parameter,k_index,encoder_vanillas_shape,encoder_futures_shape,decoder_shape):
    super(NN, self).__init__()

    # Shapes
    self.k_vanilla = np.copy(k_vanilla)
    self.k_index = np.copy(k_index)
    self.shape_vanilla = np.shape(k_vanilla)
    self.shape_future = shape_future
    self.shape_parameter = shape_parameter
    self.shape_output = np.shape(k_index)
    
    # Arquitectures
    self.encoder_vanillas_shape = copy.deepcopy(encoder_vanillas_shape)
    self.encoder_futures_shape = copy.deepcopy(encoder_futures_shape)
    self.decoder_shape = copy.deepcopy(decoder_shape)
    assert (self.encoder_vanillas_shape[-1]+self.encoder_futures_shape[-1]+self.shape_parameter[0] == self.decoder_shape[0]), "ERROR: the encoder and decoder shapes must match."
    assert (np.prod(self.shape_vanilla) == self.encoder_vanillas_shape[0]), "ERROR: the first layer in the vanilla encoder does not match the size of the vanillas"
    assert (np.prod(self.shape_future) == self.encoder_futures_shape[0]), "ERROR: the first layer in the futures encoder does not match the size of the futures"
    assert (np.prod(self.shape_output) == self.decoder_shape[-1]), "ERROR: the output shapes must match."

    # Input
    input_vanillas = tf.keras.Input(shape = self.shape_vanilla)
    input_futures = tf.keras.Input(shape = self.shape_future)
    input_parameters = tf.keras.Input(shape = self.shape_parameter)

    # Encoder vanillas
    encoder_vanillas = tf.keras.layers.Flatten()(input_vanillas)
    for i in range(len(self.encoder_vanillas_shape)-1):
        encoder_vanillas = tf.keras.layers.Dense(self.encoder_vanillas_shape[i], activation = "relu")(encoder_vanillas)
    encoder_vanillas = tf.keras.layers.Dense(self.encoder_vanillas_shape[-1], activation = "sigmoid")(encoder_vanillas)
    
    # Encoder futures
    encoder_futures = tf.keras.layers.Flatten()(input_futures)
    for i in range(len(self.encoder_futures_shape)-1):
        encoder_futures = tf.keras.layers.Dense(self.encoder_futures_shape[i], activation = "relu")(encoder_futures)
    encoder_futures = tf.keras.layers.Dense(self.encoder_futures_shape[-1], activation = "sigmoid")(encoder_futures)

    # Joint layer
    joint_layer = tf.keras.layers.concatenate([encoder_vanillas,encoder_futures,input_parameters])

    # Decoder layers 
    decoder = tf.keras.layers.Dense(self.decoder_shape[0],activation = "relu")(joint_layer)
    for i in range(1,len(self.decoder_shape)):
        decoder = tf.keras.layers.Dense(self.decoder_shape[i], activation = "relu")(decoder)

    decoder = tf.keras.layers.Dense(self.shape_output[0]*self.shape_output[1], activation = "linear")(decoder)
    decoder = tf.keras.layers.Reshape(self.shape_output)(decoder)

    self.model = tf.keras.models.Model(inputs=[input_vanillas, input_futures,input_parameters], outputs=[decoder])
    
class FNNVanillaFutureParameter(NN):
  def __init__(self,k_vanilla,shape_future,shape_parameter,k_index,shape,activation = "relu"):
    super(NN, self).__init__()

    # Shapes
    self.k_vanilla = np.copy(k_vanilla)
    self.k_index = np.copy(k_index)
    self.shape_vanilla = np.shape(k_vanilla)
    self.shape_future = shape_future
    self.shape_parameter = shape_parameter
    self.shape_output = np.shape(k_index)
    
    # Arquitectures
    self.shape = copy.deepcopy(shape)
    assert (np.prod(self.shape_vanilla)+np.prod(self.shape_future)+np.prod(self.shape_parameter) == self.shape[0]), "ERROR: the input shapes must match."
    assert (np.prod(self.shape_output) == self.shape[-1]), "ERROR: the output shapes must match."

    # Input
    input_vanillas = tf.keras.Input(shape = self.shape_vanilla)
    input_futures = tf.keras.Input(shape = self.shape_future)
    input_parameters = tf.keras.Input(shape = self.shape_parameter)

    # Flatten
    flatten_vanillas = tf.keras.layers.Flatten()(input_vanillas)
    flatten_futures = tf.keras.layers.Flatten()(input_futures)
    flatten_parameters = tf.keras.layers.Flatten()(input_parameters)

    # Concatenate
    joint_layer = tf.keras.layers.concatenate([flatten_vanillas,flatten_futures,flatten_parameters])


    # FNN
    for i in range(len(self.shape)-1):
        joint_layer = tf.keras.layers.Dense(self.shape[i], activation = activation)(joint_layer)
    joint_layer = tf.keras.layers.Dense(self.shape[-1], activation = "linear")(joint_layer)

    # Reshape
    joint_layer = tf.keras.layers.Reshape(self.shape_output)(joint_layer)

    # Define model
    self.model = tf.keras.models.Model(inputs=[input_vanillas, input_futures,input_parameters], outputs=[joint_layer])
    

