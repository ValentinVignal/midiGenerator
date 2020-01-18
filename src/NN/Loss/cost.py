"""
File containing the cost/reward functions
"""
import tensorflow as tf

math = tf.math


def scale_loss(y_true_a, y_pred_a):
    """

    :param y_true_a: activation, no loss (batch, nb_steps, step_size, input_size)
    :param y_pred_a: activation, no loss (batch, nb_steps, step_size, input_size)
    :return:
    """
    # projection
    true_projection = tf.reduce_sum(y_true_a, axis=(1, 2), keepdims=True)      # (batch, 1, 1, input_size)
    pred_projection = tf.reduce_sum(y_pred_a, axis=(1, 2), keepdims=True)      # (batch, 1, 1, input_size)
    # on scale
    input_size = true_projection.shape[3]
    scale_projector = [i % 12 for i in range(input_size)]
    true_scale_projection = math.unsorted_segment_sum(
        data=true_projection,
        segment_ids=scale_projector,
        num_segments=min(12, input_size)
    )       # (batch, 1, 1, 12)
    pred_scale_projection = math.unsorted_segment_sum(
        data=pred_projection,
        segment_ids=scale_projector,
        num_segments=min(12, input_size)
    )       # (batch, 1, 1, 12)
    # Mean (of non zero)
    true_sum = tf.reduce_sum(true_scale_projection, axis=3, keep_dims=True)       # (batch, 1, 1, 1)
    # Loss
    loss = pred_scale_projection * (1 - 2 * (true_scale_projection / true_sum))     # (batch, 1, 1, 12)
    loss = tf.reduce_sum(loss, axis=(1, 2, 3))      # (batch,)
    return tf.reduce_mean(loss)


def rhythm_loss(y_true_a, y_pred_a):
    """

    :param y_true_a: activation, no loss (batch, nb_steps, step_size, input_size)
    :param y_pred_a: activation, no loss (batch, nb_steps, step_size, input_size)
    :return:
    """
    # projection
    true_projection = tf.reduce_sum(y_true_a, axis=3)      # (batch, nb_steps, step_size)
    pred_projection = tf.reduce_sum(y_pred_a, axis=3)      # (batch, nb_steps, step_size)
    # Mean (of non zero)
    true_sum = tf.reduce_sum(true_projection, axis=1)       # (batch,)
    # Loss
    loss = pred_projection * (2 * (true_projection / true_sum) - 1)     # (batch,)
    return tf.reduce_sum(loss)


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

