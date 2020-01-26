"""
File containing the cost/reward functions
"""
import tensorflow as tf

math = tf.math
K = tf.keras.backend

from . import utils
from src import GlobalVariables as g


def scale_loss(y_true_a, y_pred_a, cost_value=g.loss.l_scale_cost, max_reward=None):
    """

    :param max_reward:
    :param cost_value:
    :param y_true_a: activation, no loss (batch, nb_steps, step_size, input_size)
    :param y_pred_a: activation, no loss (batch, nb_steps, step_size, input_size)
    :return:
    """
    # projection
    true_projection = tf.reduce_sum(y_true_a, axis=2, keepdims=True)  # (batch, nb_steps, 1, input_size)
    pred_projection = tf.reduce_sum(y_pred_a, axis=2, keepdims=True)  # (batch, nb_steps, 1, input_size)
    # on scale
    input_size = K.shape(true_projection)[-1]  # nb of notes used
    scale_projector = math.mod(tf.range(0, input_size), 12)  # [0, 1, 2, ..., 10, 11, 0, 1, ...]
    num_segments = math.reduce_max(scale_projector) + 1  # min(12, input_size)
    true_scale_projection = tf.transpose(
        math.unsorted_segment_sum(  # unsorted sum works only on the 1st axis
            data=tf.transpose(true_projection, perm=[3, 1, 2, 0]),  # (input_size, nb_steps, 1, batch)
            segment_ids=scale_projector,
            num_segments=num_segments
        ),
        perm=[3, 1, 2, 0]
    )  # (batch, nb_steps, 1, 12)
    pred_scale_projection = tf.transpose(
        math.unsorted_segment_sum(  # unsorted sum works only on the 1st axis
            data=tf.transpose(pred_projection, perm=[3, 1, 2, 0]),  # (input_size, nb_steps, 1, batch)
            segment_ids=scale_projector,
            num_segments=num_segments
        ),
        perm=[3, 1, 2, 0]
    )  # (batch, nb_steps, 1, 12)

    if max_reward is None:
        w = 1 / tf.reduce_sum(true_scale_projection, axis=3, keepdims=True)  # (batch, nb_steps, 1, 1) (can be nan)
    else:
        w = max_reward / math.reduce_max(true_scale_projection, axis=3,
                                         keepdims=True)  # (batch, nb_steps, 1, 1) (can be nan)
    cost_reward = - true_scale_projection * w  # (batch, nb_steps, 1, 1)  (can be nan)
    cost_reward = utils.replace_value(tensor=cost_reward,
                                      old_value=float(0),
                                      new_value=float(cost_value))  # (batch, nb_steps, 1, 1)  (can be nan)
    # If w is nan, it means there were no notes in the step, then, there is no loss or reward to add for this step
    cost_reward = utils.non_nan(with_nan=cost_reward, var_to_change=cost_reward)

    # Loss
    loss = pred_scale_projection * cost_reward
    loss = tf.reduce_sum(loss, axis=(1, 2, 3))  # (batch,)
    return tf.reduce_mean(loss)


def rhythm_loss(y_true_a, y_pred_a, cost_value=g.loss.l_rhythm_cost, max_reward=None,
                take_all_steps_rhythm=g.loss.take_all_step_rhythm):
    """

    :param take_all_steps_rhythm:
    :param max_reward:
    :param cost_value:
    :param y_true_a: activation, no loss (batch, nb_steps, step_size, input_size)
    :param y_pred_a: activation, no loss (batch, nb_steps, step_size, input_size)
    :return:
    """
    # projection
    sum_axis = (1, 3) if take_all_steps_rhythm else 3
    # (if take_all_steps_axis then nb_steps = 1 after these 2 lines)
    true_projection = tf.reduce_sum(y_true_a, axis=sum_axis, keepdims=True)  # (batch, nb_steps, step_size, 1)
    pred_projection = tf.reduce_sum(y_pred_a, axis=sum_axis, keepdims=True)  # (batch, nb_steps, step_size, 1)
    # on scale
    if max_reward is None:
        w = 1 / tf.reduce_sum(true_projection, axis=2, keepdims=True)  # (batch, nb_steps, 1, 1) (can be nan)
    else:
        w = max_reward / math.reduce_max(true_projection, axis=2,
                                         keepdims=True)  # (batch, nb_steps, 1, 1) (can be nan)
    cost_reward = - true_projection * w  # (batch, nb_steps, 1, 1)  (can be nan)
    cost_reward = utils.replace_value(tensor=cost_reward,
                                      old_value=float(0),
                                      new_value=float(cost_value))  # (batch, nb_steps, 1, 1)  (can be nan)
    # If w is nan, it means there were no notes in the step, then, there is no loss or reward to add for this step
    cost_reward = utils.non_nan(with_nan=cost_reward, var_to_change=cost_reward)

    # Los
    loss = pred_projection * cost_reward
    loss = tf.reduce_sum(loss, axis=(1, 2, 3))  # (batch,)
    return tf.reduce_mean(loss)


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
