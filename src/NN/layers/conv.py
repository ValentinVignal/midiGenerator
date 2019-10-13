import tensorflow as tf
import src.global_variables as g
import math

K = tf.keras.backend
layers = tf.keras.layers


class ConvBlock3D(layers.Layer):
    def __init__(self, filters, strides=(1, 1, 1), dropout=g.dropout):
        self.strides = strides
        self.conv = layers.Conv3D(filters=filters,
                                  kernel_size=(1, 5, 5),
                                  strides=strides,
                                  padding='same')
        self.batch_norm = layers.BatchNormalization()
        self.leaky_relu = layers.LeakyReLU()
        self.dropout = layers.Dropout(dropout)
        super(ConvBlock3D, self).__init__()

    def build(self, input_shape):
        self.conv.build(input_shape)
        new_shape = self.conv.compute_output_shape(input_shape)
        self.batch_norm.build(new_shape)
        self.leaky_relu.build(new_shape)
        self.dropout.build(new_shape)
        self._trainable_weights = self.conv.trainable_weights + self.batch_norm.trainable_weights
        self._non_trainable_weights = self.batch_norm.non_trainable_weights
        # TODO Verify there is no need to consider non_trainable_variable and trainable_variable
        super(ConvBlock3D, self).build(input_shape)

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.batch_norm(x)
        x = self.leaky_relu(x)
        return self.dropout(x)

    def compute_output_shape(self, input_shape):
        return self.conv.compute_output_shape(input_shape)


class ConvTransposedBlock3D(layers.Layer):
    def __init__(self, filters, strides=(1, 1, 1), dropout=g.dropout, final_shape=None):
        self.conv_transposed = layers.Conv3DTranspose(filters=filters,
                                                      kernel_size=(1, 5, 5),
                                                      padding='same',
                                                      strides=strides)
        self.batch_norm = layers.BatchNormalization()
        self.leaky_relu = layers.LeakyReLU()
        self.dropout = layers.Dropout(dropout)
        super(ConvTransposedBlock3D, self).__init__()

        self.final_shape = ConvTransposedBlock3D.init_final_shape(final_shape)

    @staticmethod
    def init_final_shape(final_shape):
        if final_shape is None or len(final_shape) == 5:
            return final_shape
        else:
            return (None, *final_shape)

    def build(self, input_shape):
        self.conv_transposed.build(input_shape)
        if self.final_shape is None:
            new_shape = self.conv_transposed.compute_output_shape(input_shape)
        else:
            new_shape = self.final_shape
        self.batch_norm.build(new_shape)
        self.leaky_relu.build(new_shape)
        self.dropout.build(new_shape)
        self._trainable_weights = self.conv_transposed.trainable_weights + self.batch_norm.trainable_weights
        self._non_trainable_weights = self.batch_norm.non_trainable_weights
        # TODO Verify there is no need to consider non_trainable_variable and trainable_variable
        super(ConvTransposedBlock3D, self).build(input_shape)

    def call(self, inputs):
        x = self.conv_transposed(inputs)
        if self.final_shape is not None:
            if x.shape[3] != self.final_shape[3]:
                x = x[:, :, :, :-1]
            if x.shape[2] != self.final_shape[2]:
                x = x[:, :, :-1]
        x = self.batch_norm(x)
        x = self.leaky_relu(x)
        return self.dropout(x)

    def compute_output_shape(self, input_shape):
        if self.final_shape is not None:
            return self.final_shape
        else:
            return self.conv_transposed.compute_output_shape(input_shape)


def new_shape_conv(input_shape, strides, filters):
    """
    To handle with padding = 'same'
    number of dim = len(input_shapes) (without batch)
    :param input_shape:
    :param strides:
    :param filters:
    :return:
    """
    new_shape = []
    for dim, stride in zip(input_shape[:-1], strides):
        new_shape.append(math.ceil(dim / stride))
    new_shape.append(filters)
    return tuple(new_shape)


def new_shapes_conv(input_shape, strides_list, filters_list):
    """
    Use to find the output shapes of several convolutional layers
    :param input_shape:
    :param strides_list:
    :param filters_list:
    :return:
    """
    print('input shape new shape conv', input_shape)
    new_shapes = [input_shape]
    for strides, filters in zip(strides_list, filters_list):
        new_shapes.append(new_shape_conv(new_shapes[-1], strides, filters))
    return new_shapes


def reverse_conv_param(original_dim, param_list):
    """

    ----------
    (nb_instrument * 2 ->) [[a, b], [c, d, e]] has to become (e ->) [[d, c, b], [a, nb_instruments]]
    And the UpSampling is done on the first convolution

    To do so:
        (1)     [[a, b]] , [c, d, e]]       <-- param_list
        (2)     [a, b, c, d, e]       # dims = [2, 3]
        (3)     [e, d, c, b, a]       # dims = [3, 2]
        (3)     [d, c, b, a , original_dim]       # save dims = [3, 2]
        (4)     [[d, c, b], [a, original_dim]]     <-- reversed_param_list
    ----------

    :param original_dim:
    :param param_list: ex [[a, b], [c, d, e]]
    :return: ex [[d, c, b], [a, original_dim]]
    """

    reversed_param_list_dims = [len(sublist) for sublist in param_list]
    reversed_param_list_temp = [size for sublist in param_list for size in sublist]  # Flatten the 2-level list
    reversed_param_list_temp = reversed_param_list_temp[::-1]  # Reversed
    reversed_param_list_dims = reversed_param_list_dims[::-1]
    reversed_param_list_temp = reversed_param_list_temp[1:] + [original_dim]  # Update shapes
    reversed_param_list = []  # Final reversed_param_list parameters
    offset = 0
    for sublist_size in reversed_param_list_dims:
        reversed_param_list.append(reversed_param_list_temp[offset: offset + sublist_size])
        offset += sublist_size
    print('reversed_param list', reversed_param_list)
    return reversed_param_list