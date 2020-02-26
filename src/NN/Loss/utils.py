import tensorflow as tf

math = tf.math
Lambda = tf.keras.layers.Lambda
K = tf.keras.backend


def get_activation(x, activation_indice=-1):
    return Lambda(lambda x: tf.gather(x, axis=activation_indice, indices=0))(x)


def get_duration(x, duration_indice=-1):
    return Lambda(lambda x: tf.gather(x, axis=duration_indice, indices=1))(x)


def non_nan(with_nan, var_to_change):
    return tf.where(math.is_nan(with_nan), tf.zeros_like(var_to_change), var_to_change)


def replace_value(tensor, old_value, new_value):
    return tf.where(tf.equal(old_value, tensor), new_value * tf.ones_like(tensor), tensor)


def to_scale(tensor, axis=-1):
    """

    :param axis: Axis to to the transformation
    :param tensor: whatever shape, in example: (batch, nb_instrument, nb_steps, step_size, input_size)
    :return: same shape but with max if axis dim = 12
    """
    nb_axis = tf.rank(tensor)  # Number of dimension in the tensor (=5)
    input_size = K.shape(tensor)[axis]  # nb of notes used
    absolute_axis = axis if axis >= 0 else nb_axis + axis  # absolute_axis >= 0
    scale_projector = math.mod(tf.range(0, input_size), 12)  # [0, 1, 2, ..., 10, 11, 0, 1, ...]
    num_segments = math.reduce_max(scale_projector) + 1  # min(12, input_size)
    first_transpose_axis = tf.roll(tf.range(nb_axis), shift=-absolute_axis, axis=0)     # [4, 0, 1, 2, 3]
    second_transpose_axis = tf.roll(tf.range(nb_axis), shift=absolute_axis, axis=0)     # [1, 2, 3, 4, 0]

    scale = tf.transpose(
        math.unsorted_segment_sum(  # unsorted sum works only on the 1st axis
            data=tf.transpose(tensor, perm=first_transpose_axis),
            # (input_size, 1=nb_instruments, nb_steps, 1=step_size, batch)
            segment_ids=scale_projector,
            num_segments=num_segments
        ),  # (12, nb_instruments, nb_steps, step_size, batch)
        perm=second_transpose_axis
    )  # (batch, 1=nb_instruments, nb_steps, 1=step_size, 12)

    return scale
