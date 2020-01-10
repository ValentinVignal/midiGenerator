import tensorflow as tf

Lambda = tf.keras.layers.Lambda
math = tf.math


def acc_act(y_true, y_pred):
    y_true_a = Lambda(lambda x: x[:, :, :, 0])(y_true)
    y_pred_a = Lambda(lambda x: x[:, :, :, 0])(y_pred)

    acc = tf.keras.metrics.binary_accuracy(y_true_a, y_pred_a)

    return tf.reduce_mean(acc, axis=None)


def mae_dur(y_true, y_pred):
    y_true_d = Lambda(lambda x: x[:, :, :, 1])(y_true)
    y_pred_d = Lambda(lambda x: x[:, :, :, 1])(y_pred)

    mae = tf.keras.metrics.mean_squared_error(y_true_d, y_pred_d)

    return tf.reduce_mean(mae, axis=None)


def acc_mono():
    def acc(y_true, y_pred):
        y_true_a = Lambda(lambda x: tf.gather(x, axis=4, indices=0))(y_true)
        y_pred_a = Lambda(lambda x: tf.gather(x, axis=4, indices=0))(y_pred)
        y_true_a_no_nan = tf.where(math.is_nan(y_true_a), tf.zeros_like(y_true_a), y_true_a)
        y_pred_a_no_nan = tf.where(math.is_nan(y_true_a), tf.zeros_like(y_pred_a), y_pred_a)     # To apply loss only on non nan value in true Tensor

        acc = tf.keras.metrics.categorical_accuracy(y_true_a_no_nan, y_pred_a_no_nan)

        return tf.reduce_mean(acc, axis=None)
    return acc

