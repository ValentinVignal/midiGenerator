"""
File containing all the usable cost function
"""
import tensorflow as tf

from . import utils
from . import cost

K = tf.keras.backend
math = tf.math
Lambda = tf.keras.layers.Lambda

from src import GlobalVariables as g


def basic(lambda_a, lambda_d, *args, **kwargs):
    def _basic(y_true, y_pred):
        """

        :param y_true: (batch, lenght, input_size, 2)
        :param y_pred: (batch, lenght, input_size, 2)
        :return:
        """
        y_true_a = utils.get_activation(y_true, activation_indice=4)
        y_true_d = utils.get_duration(y_true, duration_indice=4)
        y_pred_a = utils.get_activation(y_pred, activation_indice=4)
        y_pred_d = utils.get_duration(y_pred, duration_indice=4)

        loss_a = tf.keras.losses.binary_crossentropy(y_true_a, y_pred_a)
        loss_d = tf.keras.losses.mean_squared_error(y_true_d, y_pred_d)

        loss = lambda_a * loss_a + lambda_d * loss_d

        return tf.reduce_mean(loss, axis=None)

    return _basic


def mono(*args, **kwargs):
    """
    Used for model which can predict only one note at the same time
    :param args:
    :param kwargs:
    :return:
    """

    def _mono(y_true, y_pred):
        """
        y_pred has np nan where we shouldn't compute the loss

        :param y_true: (batch, nb_steps=1, step_size, input_size, channels=1)
        :param y_pred: (batch, nb_steps=1, step_size, input_size, channels=1)
        :return:
        """
        y_true_a = utils.get_activation(y_true)
        y_pred_a = utils.get_activation(y_pred)
        y_pred_a_no_nan = utils.non_nan(with_nan=y_true_a, var_to_change=y_pred_a)
        y_true_a_no_nan = utils.non_nan(with_nan=y_true_a, var_to_change=y_true_a)

        loss = tf.keras.losses.categorical_crossentropy(y_true_a_no_nan, y_pred_a_no_nan)
        loss = tf.where(math.is_nan(loss), tf.zeros_like(loss), loss)
        loss = tf.reduce_sum(loss, axis=(1, 2))
        return tf.reduce_mean(loss)

    return _mono


def mono_scale(l_scale=g.loss.l_scale, l_rhythm=g.loss.l_rhythm, l_scale_cost=g.loss.l_scale_cost,
               l_rhythm_cost=g.loss.l_rhythm_cost, take_all_steps_rhythm=g.loss.take_all_step_rhythm,
               *args, **kwargs):
    """
    Add the scale and rhythm reward/cost
    :param take_all_steps_rhythm:
    :param l_rhythm_cost:
    :param l_scale_cost:
    :param l_scale:
    :param l_rhythm:
    :param args:
    :param kwargs:
    :return:
    """

    def _mono_scale(y_true, y_pred):
        y_true_a = utils.get_activation(y_true)
        y_pred_a = utils.get_activation(y_pred)
        y_pred_a = utils.non_nan(with_nan=y_true_a, var_to_change=y_pred_a)
        y_true_a = utils.non_nan(with_nan=y_true_a, var_to_change=y_true_a)

        loss = mono()(y_true, y_pred)
        loss += l_scale * cost.scale_loss(y_true_a[:, :-1], y_pred_a[:, :-1],
                                          cost_value=l_scale_cost, has_instruments_dim=False)
        loss += l_rhythm * cost.rhythm_loss(y_true_a, y_pred_a,
                                            cost_value=l_rhythm_cost,
                                            take_all_steps_rhythm=take_all_steps_rhythm, has_instruments_dim=False)
        return loss

    return _mono_scale


def scale(l_scale=g.loss.l_scale, l_rhythm=g.loss.l_rhythm, l_scale_cost=g.loss.l_scale_cost,
          l_rhythm_cost=g.loss.l_rhythm_cost, take_all_steps_rhythm=g.loss.take_all_step_rhythm,
          is_mono=False,
          *args, **kwargs):
    """
    Add the scale and rhythm reward/cost
    :param take_all_steps_rhythm:
    :param l_rhythm_cost:
    :param l_scale_cost:
    :param l_scale:
    :param l_rhythm:
    :param args:
    :param kwargs:
    :return:
    """

    def _scale(y_true, y_pred):
        """

        :param y_true: (batch, nb_instruments, nb_steps, step_size, input_size, channels)
        :param y_pred: (batch, nb_instruments, nb_steps, step_size, input_size, channels)
        :return:
        """
        y_true_a = utils.get_activation(y_true)
        y_pred_a = utils.get_activation(y_pred)
        if is_mono:
            y_true_a = y_true_a[:, :-1]
            y_pred_a = y_pred_a[:, :-1]
        # y_a: (batch, nb_instruments, nb_steps, step_size, input_size)
        y_pred_a = utils.non_nan(with_nan=y_true_a, var_to_change=y_pred_a)
        y_true_a = utils.non_nan(with_nan=y_true_a, var_to_change=y_true_a)

        loss = l_scale * cost.scale_loss(y_true_a, y_pred_a,
                                         cost_value=l_scale_cost)
        loss += l_rhythm * cost.rhythm_loss(y_true_a, y_pred_a,
                                            cost_value=l_rhythm_cost,
                                            take_all_steps_rhythm=take_all_steps_rhythm)
        return loss

    return _scale
