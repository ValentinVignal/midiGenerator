"""
File containing the cost/reward functions
"""
import tensorflow as tf

math = tf.math
K = tf.keras.backend

from . import utils
from src import GlobalVariables as g


def scale(y_true_a, y_pred_a, cost_value=g.loss.l_scale_cost, max_reward=None):
    """

    :param max_reward:
    :param cost_value:
    :param y_true_a: activation, no loss (batch, nb_instruments, nb_steps, step_size, input_size)
    :param y_pred_a: activation, no loss (batch, nb_instruments, nb_steps, step_size, input_size)
    :return:
    """
    # projection
    true_projection = tf.reduce_sum(y_true_a, axis=[1, 3], keepdims=True)  # (batch, 1, nb_steps, 1, input_size)
    pred_projection = tf.reduce_sum(y_pred_a, axis=[1, 3], keepdims=True)  # (batch, 1, nb_steps, 1, input_size)
    # on scale
    true_scale_projection = utils.to_scale(true_projection, axis=-1)
    pred_scale_projection = utils.to_scale(pred_projection, axis=-1)

    if max_reward is None:
        w = 1 / tf.reduce_sum(true_scale_projection, axis=4, keepdims=True)  # (batch, 1, nb_steps, 1, 1) (can be nan)
    else:
        w = max_reward / math.reduce_max(true_scale_projection, axis=4,
                                         keepdims=True)  # (batch, 1, nb_steps, 1, 1) (can be nan)
    cost_reward = - true_scale_projection * w  # (batch, 1, nb_steps, 1, 1)  (can be nan)
    cost_reward = utils.replace_value(tensor=cost_reward,
                                      old_value=float(0),
                                      new_value=float(cost_value))  # (batch, 1, nb_steps, 1, 1)  (can be nan)
    # If w is nan, it means there were no notes in the step, then, there is no loss or reward to add for this step
    cost_reward = utils.non_nan(with_nan=cost_reward, var_to_change=cost_reward)

    # Loss
    loss = pred_scale_projection * cost_reward
    loss = tf.reduce_sum(loss, axis=(1, 2, 3, 4))  # (batch,)
    return tf.reduce_mean(loss)


def rhythm(y_true_a, y_pred_a, cost_value=g.loss.l_rhythm_cost, max_reward=None,
           take_all_steps_rhythm=g.loss.take_all_step_rhythm):
    """

    :param take_all_steps_rhythm:
    :param max_reward:
    :param cost_value:
    :param y_true_a: activation, no loss (batch, nb_instruments, nb_steps, step_size, input_size)
    :param y_pred_a: activation, no loss (batch, nb_instruments, nb_steps, step_size, input_size)
    :return:
    """
    # projection
    sum_axis = (1, 2, 4) if take_all_steps_rhythm else (1, 4)
    # (if take_all_steps_axis then nb_steps = 1 after these 2 lines)
    true_projection = tf.reduce_sum(y_true_a, axis=sum_axis, keepdims=True)  # (batch, 1, nb_steps, step_size, 1)
    pred_projection = tf.reduce_sum(y_pred_a, axis=sum_axis, keepdims=True)  # (batch, 1, nb_steps, step_size, 1)
    # on scale
    if max_reward is None:
        w = 1 / tf.reduce_sum(true_projection, axis=3,
                              keepdims=True)  # (batch, 1, nb_steps, 1=step_size, 1) (can be nan)
    else:
        w = max_reward / math.reduce_max(true_projection, axis=3,
                                         keepdims=True)  # (batch, 1, nb_steps, 1, 1) (can be nan)
    cost_reward = - true_projection * w  # (batch, 1, nb_steps, 1, 1)  (can be nan)
    cost_reward = utils.replace_value(tensor=cost_reward,
                                      old_value=float(0),
                                      new_value=float(cost_value))  # (batch, 1, nb_steps, 1, 1)  (can be nan)
    # If w is nan, it means there were no notes in the step, then, there is no loss or reward to add for this step
    cost_reward = utils.non_nan(with_nan=cost_reward, var_to_change=cost_reward)

    # Los
    loss = pred_projection * cost_reward
    loss = tf.reduce_sum(loss, axis=(1, 2, 3, 4))  # (batch,)
    return tf.reduce_mean(loss)


def harmony(y_true_a, y_pred_a, interval):
    """

    :param y_true_a: activation, (batch, nb_instruments, nb_steps, step_size, input_size) (no mono silent note)
    :param y_pred_a: activation, (batch, nb_instruments, nb_steps, step_size, input_size) (no mono silent note)
    :param interval:
    :return:
    """
    y_true_scale = utils.to_scale(y_true_a, axis=-1)        # (batch, nb_instruments, nb_steps, step_size, 12)
    y_pred_scale = utils.to_scale(y_pred_a, axis=-1)        # (batch, nb_instruments, nb_steps, step_size, 12)






# --------------------------------------------------
# ------------------- KL Divergence --------------------
# --------------------------------------------------

def kld(mean, std, sum_axis=None):
    """

    :param sum_axis: axis to sum, if None, there is no sum
    :param mean:
    :param std:
    :return:
    """

    res = - 0.5 * (2 * math.log(std) - math.square(mean) - math.square(std) + 1)
    if sum_axis is not None:
        res = tf.reduce_sum(res, axis=sum_axis)
    res = tf.reduce_mean(res)
    return res
