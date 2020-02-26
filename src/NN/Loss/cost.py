"""
File containing the cost/reward functions
"""
import tensorflow as tf

math = tf.math
K = tf.keras.backend

from . import utils
from src import GlobalVariables as g


# --------------------------------------------------
# ------------------- Scale  --------------------
# --------------------------------------------------


def scale(y_true_a, y_pred_a):
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
    true_scale_projection = utils.to_scale(
        true_projection, axis=-1
    )        # (batch, 1=nb_instruments, nb_steps, 1=step_length, 12)
    pred_scale_projection = utils.to_scale(
        pred_projection, axis=-1
    )        # (batch, 1=nb_instruments, nb_steps, 1=step_length, 12)

    nb_zeros = utils.count(
        true_scale_projection, 0, axis=[-1], keepdims=True
    )       # (batch, 1=nb_instruments, nb_steps, 1=step_length, 1)
    cost_value = 1 / nb_zeros       # Nan/inf values won't be used

    w = 1 / tf.reduce_sum(true_scale_projection, axis=4, keepdims=True)  # (batch, 1, nb_steps, 1, 1) (can be inf)
    # If there is an inf value, it means there is no note played in truth
    # No note played in truth -> Can't construct a scale -> no loss

    cost_reward = - true_scale_projection * w  # (batch, 1, nb_steps, 1, 1)  (can be nan)
    cost_reward = tf.where(tf.equal(float(0), cost_reward), tf.cast(cost_value, tf.float32), cost_reward)
    # If w is nan, it means there were no notes in the step, then, there is no loss or reward to add for this step
    cost_reward = utils.non_inf(with_nan=cost_reward, var_to_change=cost_reward)
    cost_reward = utils.non_nan(with_nan=cost_reward, var_to_change=cost_reward)

    # Loss
    loss = pred_scale_projection * cost_reward
    loss = tf.reduce_sum(loss, axis=(1, 2, 3, 4))  # (batch,)
    return tf.reduce_mean(loss)


# --------------------------------------------------
# ------------------- Rhythm --------------------
# --------------------------------------------------


def rhythm(y_true_a, y_pred_a,
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

    nb_zeros = utils.count(
        true_projection, 0, axis=[-2], keepdims=True
    )       # (batch, 1=nb_instruments, nb_steps, 1=step_length, 1)
    cost_value = 1 / nb_zeros       # Nan/inf values won't be used

    # on rhythm
    w = 1 / tf.reduce_sum(true_projection, axis=3,
                          keepdims=True)  # (batch, 1, nb_steps, 1=step_size, 1) (can be nan)
    cost_reward = - true_projection * w  # (batch, 1, nb_steps, 1, 1)  (can be nan)
    cost_reward = tf.where(tf.equal(float(0), cost_reward), tf.cast(cost_value, tf.float32), cost_reward)
    # If w is nan, it means there were no notes in the step, then, there is no loss or reward to add for this step
    cost_reward = utils.non_nan(with_nan=cost_reward, var_to_change=cost_reward)
    cost_reward = utils.non_inf(with_nan=cost_reward, var_to_change=cost_reward)

    # Los
    loss = pred_projection * cost_reward
    loss = tf.reduce_sum(loss, axis=(1, 2, 3, 4))  # (batch,)
    return tf.reduce_mean(loss)


# --------------------------------------------------
# ------------------- Harmony --------------------
# --------------------------------------------------


def n_tone(tensor, interval):
    """

    :param tensor: activation, (batch, nb_instruments, nb_steps, step_size, input_size) (no mono silent note)
    :param interval:
    :return:
    """
    tensor = utils.to_scale(tensor, axis=-1)        # (batch, nb_instruments, nb_steps, step_size, 12)
    tensor = math.reduce_sum(tensor, axis=(1, 2, 3))        # (batch, 12)
    tensor_shift = tf.roll(tensor, shift=interval, axis=1)      # (batch, 12)
    # Scalar product
    loss = math.reduce_sum(tensor * tensor_shift, axis=1)       # batch
    return math.reduce_mean(loss)


def semitone(tensor):
    """

    :param tensor:
    :return:
    """
    return n_tone(tensor, interval=1)


def tone(tensor):
    """

    :param tensor:
    :return:
    """
    return n_tone(tensor, interval=2)


def tritone(tensor):
    """

    :param tensor:
    :return:
    """
    return n_tone(tensor, interval=6)


def harmony(*args, l_semitone=g.loss.l_semitone, l_tone=g.loss.l_tone, l_tritone=g.loss.l_tritone, **kwargs):
    """

    :param args:
    :param l_semitone:
    :param l_tone:
    :param l_tritone:
    :param kwargs:
    :return:
    """
    def _harmony(tensor):
        """

        :param tensor:
        :return:
        """
        loss = l_semitone * semitone(tensor)
        loss += l_tone * tone(tensor)
        loss += l_tritone * tritone(tensor)
        return loss

    return _harmony


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
