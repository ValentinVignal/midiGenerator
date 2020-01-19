import tensorflow as tf
import numpy as np
K = tf.keras.backend
math = tf.math

Lambda = tf.keras.layers.Lambda

# --------------------------------------------------
# -------------------- Operations --------------------
# --------------------------------------------------
def get_activation(x):
    return Lambda(lambda x: tf.gather(x, axis=4, indices=0))(x)


def get_duration(x):
    return Lambda(lambda x: tf.gather(x, axis=4, indices=1))(x)


def non_nan(with_nan, var_to_change):
    return tf.where(math.is_nan(with_nan), tf.zeros_like(var_to_change), var_to_change)




def choose_loss(loss_name):
    """

    :param loss_name:
    :return: The corresponding loss function corresponding to the string loss_name
    """
    if loss_name == 'common':
        return loss_common
    elif loss_name == 'mono':
        return loss_mono
    elif loss_name == 'mono_scale':
        return loss_mono_scale
    else:
        raise Exception(f'type_loss "{loss_name}" not known')

# --------------------------------------------------
# -------------------- Real output --------------------
# --------------------------------------------------


def loss_common(lambda_a, lambda_d, *args, **kwargs):
    def _loss_common(y_true, y_pred):
        """

        :param y_true: (batch, lenght, input_size, 2)
        :param y_pred: (batch, lenght, input_size, 2)
        :return:
        """
        y_true_a = Lambda(lambda x: x[:, :, :, 0])(y_true)
        y_true_d = Lambda(lambda x: x[:, :, :, 1])(y_true)
        y_pred_a = Lambda(lambda x: x[:, :, :, 0])(y_pred)
        y_pred_d = Lambda(lambda x: x[:, :, :, 1])(y_pred)

        loss_a = tf.keras.losses.binary_crossentropy(y_true_a, y_pred_a)
        loss_d = tf.keras.losses.mean_squared_error(y_true_d, y_pred_d)

        loss = lambda_a * loss_a + lambda_d * loss_d

        return tf.reduce_mean(loss, axis=None)

    return _loss_common




def loss_mono(*args, **kwargs):
    def _loss_mono(y_true, y_pred):
        """
        y_pred has np nan where we shouldn't compute the loss

        :param y_true: (batch, nb_steps=1, step_size, input_size, channels=1)
        :param y_pred: (batch, nb_steps=1, step_size, input_size, channels=1)
        :return:
        """
        y_true_a = get_activation(y_true)
        y_pred_a = get_activation(y_pred)
        y_pred_a_no_nan = non_nan(with_nan=y_true_a, var_to_change=y_pred_a)
        y_true_a_no_nan = non_nan(with_nan=y_true_a, var_to_change=y_true_a)

        loss = tf.keras.losses.categorical_crossentropy(y_true_a_no_nan, y_pred_a_no_nan)
        loss = tf.where(math.is_nan(loss), tf.zeros_like(loss), loss)
        return loss
    return _loss_mono


def loss_mono_scale(l_scale, l_rythm, *args, **kwargs):
    def _loss_mono_scale(y_true, y_pred):
        y_true_a = get_activation(y_true)
        y_pred_a = get_activation(y_pred)
        y_pred_a = non_nan(with_nan=y_true_a, var_to_change=y_pred_a)
        y_true_a = non_nan(with_nan=y_true_a, var_to_change=y_true_a)

        loss = loss_mono()(y_true, y_pred)
        loss += l_scale * scale_loss(y_true_a, y_pred_a)
        loss += l_rythm * rythm_loss(y_true_a, y_pred_a)
        return loss
    return _loss_mono_scale


# --------------------------------------------------
# ------------------- Cost function --------------------
# --------------------------------------------------
# (not usable directly in NN)


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


def rythm_loss(y_true_a, y_pred_a):
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

def kld(mean, std):
    """

    :param mean:
    :param std:
    :return:
    """
    return - 0.5 * tf.reduce_mean(
        2 * math.log(std) - math.square(mean) - math.square(std) + 1
    )
