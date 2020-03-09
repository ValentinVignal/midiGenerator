import tensorflow as tf

from .. import utils

Lambda = tf.keras.layers.Lambda
math = tf.math


def acc_activation():
    def acc_act(y_true, y_pred):
        """

        :param y_true:
        :param y_pred:
        :return:
        """
        """
        y_true_a = Lambda(lambda x: x[:, :, :, 0])(y_true)
        y_pred_a = Lambda(lambda x: x[:, :, :, 0])(y_pred)
        """
        y_true_a = utils.get_activation(y_true)
        y_pred_a = utils.get_activation(y_pred)

        acc = tf.keras.metrics.binary_accuracy(y_true_a, y_pred_a)

        return tf.reduce_mean(acc, axis=None)
    return acc_act


def mae_duration():
    def mae_dur(y_true, y_pred):
        """
        mean squarred error
        :param y_true:
        :param y_pred:
        :return:
        """
        """
        y_true_d = Lambda(lambda x: x[:, :, :, 1])(y_true)
        y_pred_d = Lambda(lambda x: x[:, :, :, 1])(y_pred)
        """
        y_true_d = utils.get_duration(y_true)
        y_pred_d = utils.get_duration(y_pred)

        mae = tf.keras.metrics.mean_squared_error(y_true_d, y_pred_d)

        return tf.reduce_mean(mae, axis=None)
    return mae_dur


def acc_mono():
    def acc(y_true, y_pred):
        """
        :param y_true: (batch, nb_steps=1, step_size, input_size, channels=1)
        :param y_pred: (batch, nb_steps=1, step_size, input_size, channels=1)
        """
        y_true_a = utils.get_activation(y_true)
        y_pred_a = utils.get_activation(y_pred)
        y_pred_a_no_nan = utils.non_nan(y_true_a, y_pred_a)
        y_true_a_no_nan = utils.non_nan(y_true_a, y_true_a)

        acc_ = tf.keras.metrics.categorical_accuracy(y_true_a_no_nan, y_pred_a_no_nan)
        acc_ = tf.reduce_mean(acc_)
        return acc_
    return acc


def acc_mono_cat():
    def acc_cat(y_true, y_pred):
        """
        :param y_true: (batch, nb_steps=1, step_size, input_size, channels=1)
        :param y_pred: (batch, nb_steps=1, step_size, input_size, channels=1)
        """
        y_true_a = utils.get_activation(y_true)
        y_pred_a = utils.get_activation(y_pred)
        y_pred_a_no_nan = utils.non_nan(y_true_a, y_pred_a)[:, :, :, :-1]
        y_true_a_no_nan = utils.non_nan(y_true_a, y_true_a)[:, :, :, :-1]

        acc = tf.keras.metrics.categorical_accuracy(y_true_a_no_nan, y_pred_a_no_nan)
        acc = tf.reduce_mean(acc)
        return acc

    return acc_cat


def acc_mono_bin():
    def acc_bin(y_true, y_pred):
        """
        :param y_true: (batch, nb_steps=1, step_size, input_size, channels=1)
        :param y_pred: (batch, nb_steps=1, step_size, input_size, channels=1)
        """
        y_true_a = utils.get_activation(y_true)
        y_pred_a = utils.get_activation(y_pred)
        y_pred_a_no_nan = utils.non_nan(y_true_a, y_pred_a)[:, :, :, -1:]
        y_true_a_no_nan = utils.non_nan(y_true_a, y_true_a)[:, :, :, -1:]

        acc = tf.keras.metrics.binary_accuracy(y_true_a_no_nan, y_pred_a_no_nan)
        acc = tf.reduce_mean(acc)
        return acc

    return acc_bin
