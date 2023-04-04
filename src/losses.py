import tensorflow as tf

class MaxAbsoluteError(tf.keras.losses.Loss):
    def __init__(self):
        super().__init__()

    def call(self, y_true, y_pred):
        absolute_value = tf.math.abs(tf.math.subtract(y_true,y_pred))
        return tf.reduce_max(absolute_value,axis = 0)

class MaxAbsolutePercentageError(tf.keras.losses.Loss):
    def __init__(self):
        super().__init__()

    def call(self, y_true, y_pred):
        absolute_value = tf.math.abs(tf.math.divide(tf.math.subtract(y_true,y_pred),y_true)*100)
        return tf.reduce_max(absolute_value,axis = 0)
