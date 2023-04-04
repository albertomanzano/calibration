import tensorflow as tf

def MaxAbsoluteError(y_true,y_pred):
    absolute_value = tf.math.abs(tf.math.subtract(y_true,y_pred))
    return tf.reduce_max(absolute_value,axis = 0)

def MaxAbsolutePercentageError(y_true,y_pred):
    absolute_value = tf.math.abs(tf.math.divide(tf.math.subtract(y_true,y_pred),y_true)*100)
    return tf.reduce_max(absolute_value,axis = 0)

