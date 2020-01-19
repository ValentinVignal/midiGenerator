import tensorflow as tf

math = tf.math
Lambda = tf.keras.layers.Lambda


def get_activation(x, activation_indice=4):
    return Lambda(lambda x: tf.gather(x, axis=activation_indice, indices=0))(x)


def get_duration(x, duration_indice=4):
    return Lambda(lambda x: tf.gather(x, axis=duration_indice, indices=1))(x)


def non_nan(with_nan, var_to_change):
    return tf.where(math.is_nan(with_nan), tf.zeros_like(var_to_change), var_to_change)


def replace_value(tensor, old_value, new_value):
    return tf.where(tf.equal(old_value, tensor), new_value * tf.ones_like(tensor), tensor)
