import tensorflow as tf

class Autoencoder(tf.keras.models.Model):
  def __init__(self,shape_input,shape_output,neurons_layer):
    super(Autoencoder, self).__init__()
    self.shape_input = shape_input
    self.shape_output = shape_output
    self.neurons_layer = neurons_layer

    # Encoder
    self.encoder = tf.keras.models.Sequential()
    self.encoder.add(tf.keras.Input(shape=shape_input))
    self.encoder.add(tf.keras.layers.Flatten())
    for i in range(len(neurons_layer)-1):
        self.encoder.add(tf.keras.layers.Dense(neurons_layer[i], activation = "relu"))
    self.encoder.add(tf.keras.layers.Dense(neurons_layer[-1], activation = "sigmoid"))

    # Decoder
    self.decoder = tf.keras.models.Sequential()
    self.decoder.add(tf.keras.Input(shape=(neurons_layer[-1],)))
    for i in range(len(neurons_layer)-2,-1,-1):
        self.decoder.add(tf.keras.layers.Dense(neurons_layer[i], activation = "relu"))
    self.decoder.add(tf.keras.layers.Dense(shape_output[0]*shape_output[1], activation = "linear"))
    self.decoder.add(tf.keras.layers.Reshape(shape_output))
    

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded
