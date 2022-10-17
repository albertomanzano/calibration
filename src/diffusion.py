import tensorflow as tf

class DiffusionFCNN:

    def __init__(self,layers = 3,neurons = 3):
        self.network = tf.keras.Sequential()
        self.network.add(tf.keras.layers.Input(shape = (2,)),)
        for i in range(layers):
            self.network.add(tf.keras.layers.Dense(neurons, activation="sigmoid"))
        self.network.add(tf.keras.layers.Dense(1, activation="linear"))
        self.trainable_parameters = self.network.trainable_weights

    @tf.function
    def __call__(self,t,S):
        points = tf.stack([t,S], axis = 1)
        result = self.network(points)
        return tf.reshape(result,[-1])

class DiffusionConstant:

    def __init__(self,vol):
        self.vol = vol

    @tf.function
    def __call__(self,t,S):
        return self.vol*S
