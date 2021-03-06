import tensorflow as tf
from src import GlobalVariables as g
import math
from .KerasLayer import KerasLayer

import src.mtypes as t

K = tf.keras.backend
layers = tf.keras.layers


# ------------------------------------------------------------
#                               3D
# ------------------------------------------------------------

class ConvBlock3D(KerasLayer):
    def __init__(self, filters: int, strides: t.strides = (1, 1, 1), dropout: float = g.nn.dropout_c, *args, **kwargs):
        """

        :param filters: int: the size of the filters
        :param strides: tuple<int>: (3,):
        :param dropout: float:
        """
        super(ConvBlock3D, self).__init__(*args, **kwargs)
        # ---------- Raw parameters ----------
        self.filters = filters
        self.strides = strides
        self.dropout = dropout

        self.conv = layers.Conv3D(filters=filters,
                                  kernel_size=(1, 5, 5),
                                  strides=strides,
                                  padding='same')
        self.batch_norm = layers.BatchNormalization()
        self.leaky_relu = layers.LeakyReLU()
        self.dropout_layer = layers.Dropout(dropout)

    def get_config(self):
        config = super(ConvBlock3D, self).get_config()
        config.update(dict(
            filters=self.filters,
            strides=self.strides,
            dropout=self.dropout
        ))

    def build(self, input_shape: t.bshape):
        self.conv.build(input_shape)
        new_shape = self.conv.compute_output_shape(input_shape)
        self.batch_norm.build(new_shape)
        self.leaky_relu.build(new_shape)
        self.dropout_layer.build(new_shape)
        self.set_weights_variables(self.conv, self.batch_norm)
        super(ConvBlock3D, self).build(input_shape)

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.batch_norm(x)
        x = self.leaky_relu(x)
        return self.dropout_layer(x)

    def compute_output_shape(self, input_shape):
        return self.conv.compute_output_shape(input_shape)


class ConvTransposedBlock3D(KerasLayer):
    def __init__(self, filters: int, strides: t.strides = (1, 1, 1), dropout: float = g.nn.dropout_c,
                 final_shape: t.anyshape_ = None, *args, **kwargs):
        """

        :param filters: int:
        :param strides: Tuple[int]:
        :param dropout: float:
        :param final_shape: Optional[Tuple[int]]
            ⚠ Batch dim in the axis0 : (?, a, b, c, d) ⚠
        """
        super(ConvTransposedBlock3D, self).__init__(*args, **kwargs)
        # ---------- Raw parameters ----------
        self.filters = filters
        self.strides = strides,
        self.dropout = dropout,
        self.final_shape: t.bshape_ = ConvTransposedBlock3D.check_final_shape(final_shape)

        self.conv_transposed = layers.Conv3DTranspose(filters=filters,
                                                      kernel_size=(1, 5, 5),
                                                      padding='same',
                                                      strides=strides)
        self.batch_norm = layers.BatchNormalization()
        self.leaky_relu = layers.LeakyReLU()
        self.dropout_layer = layers.Dropout(dropout)

    def get_config(self):
        config = super(ConvTransposedBlock3D, self).get_config()
        config.update(dict(
            filters=self.filters,
            strides=self.strides,
            dropout=self.dropout,
            final_shape=self.final_shape
        ))
        return config

    @staticmethod
    def check_final_shape(final_shape: t.anyshape_) -> t.bshape_:
        """
        if batch dim is not in the shape, then put it

        :param final_shape: tuple<int>: (4,) or (5,): (a, b, c, d) or (?, a, b, c, d)
        :return: tuple<int>: (5,): (?, a, b, c, d)
        """
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
        self.dropout_layer.build(new_shape)
        self.set_weights_variables(self.conv_transposed, self.batch_norm)
        super(ConvTransposedBlock3D, self).build(input_shape)

    def call(self, inputs):
        x = self.conv_transposed(inputs)
        if self.final_shape is not None:
            if x.shape[3] != self.final_shape[3]:       # Input size check
                x = x[:, :, :, :-1]
            if x.shape[2] != self.final_shape[2]:       # step_size check
                x = x[:, :, :-1]
        x = self.batch_norm(x)
        x = self.leaky_relu(x)
        return self.dropout_layer(x)

    def compute_output_shape(self, input_shape):
        if self.final_shape is not None:
            return self.final_shape
        else:
            return self.conv_transposed.compute_output_shape(input_shape)


# ------------------------------------------------------------
#                               2D
# ------------------------------------------------------------

class ConvBlock2D(KerasLayer):
    def __init__(self, filters: int, strides: t.strides = (1, 1), dropout: float = g.nn.dropout_c, *args, **kwargs):
        """

        :param filters: int: the size of the filters
        :param strides: tuple<int>: (2,):
        :param dropout: float:
        """
        super(ConvBlock2D, self).__init__(*args, **kwargs)
        # ---------- Raw parameters ----------
        self.filters = filters
        self.strides = strides
        self.dropout = dropout

        self.conv = layers.Conv2D(filters=filters,
                                  kernel_size=(5, 5),
                                  strides=strides,
                                  padding='same')
        self.batch_norm = layers.BatchNormalization()
        self.leaky_relu = layers.LeakyReLU()
        self.dropout_layer = layers.Dropout(dropout)

    def get_config(self):
        config = super(ConvBlock2D, self).get_config()
        config.update(dict(
            fitlers=self.filters,
            strides=self.strides,
            dropout=self.dropout
        ))
        return config

    def build(self, input_shape: t.bshape):
        self.conv.build(input_shape)
        new_shape = self.conv.compute_output_shape(input_shape)
        self.batch_norm.build(new_shape)
        self.leaky_relu.build(new_shape)
        self.dropout_layer.build(new_shape)
        self.set_weights_variables(self.conv, self.batch_norm)
        super(ConvBlock2D, self).build(input_shape)

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.batch_norm(x)
        x = self.leaky_relu(x)
        x = self.dropout_layer(x)
        return x

    def compute_output_shape(self, input_shape):
        return self.conv.compute_output_shape(input_shape)


class ConvTransposedBlock2D(KerasLayer):
    def __init__(self, filters: int, strides: t.strides = (1, 1), dropout: float = g.nn.dropout_c,
                 final_shape: t.anyshape_ = None, activation=None, *args, **kwargs):
        """

        :param filters: int:
        :param strides: Tuple[int]:
        :param dropout: float:
        :param final_shape: Optional[Tuple[int]]
            ⚠ Batch dim in the axis0 : (?, a, b, c, d) ⚠
        """
        super(ConvTransposedBlock2D, self).__init__(*args, **kwargs)
        # ---------- Raw parameters ----------
        self.filters = filters
        self.strides = strides
        self.dropout = dropout
        self.final_shape = final_shape
        self.activation = 'leaky_relu' if activation is None else activation

        self.conv_transposed = layers.Conv2DTranspose(filters=filters,
                                                      kernel_size=(5, 5),
                                                      padding='same',
                                                      strides=strides)
        self.batch_norm = layers.BatchNormalization()
        if self.activation == 'leaky_relu':
            self.activation_layer = layers.LeakyReLU()
        elif self.activation == 'tanh':
            self.activation_layer = tf.keras.layers.Activation('tanh')
        self.dropout_layer = layers.Dropout(dropout)

        self.final_shape: t.bshape_ = ConvTransposedBlock2D.check_final_shape(final_shape)

    @staticmethod
    def check_final_shape(final_shape: t.anyshape_) -> t.bshape_:
        """
        if batch dim is not in the shape, then put it

        :param final_shape: tuple<int>: (4,) or (5,): (a, b, c, d) or (?, a, b, c, d)
        :return: tuple<int>: (5,): (?, a, b, c, d)
        """
        if final_shape is None or len(final_shape) == 4:
            return final_shape
        else:
            return (None, *final_shape)

    def get_config(self):
        config = super(ConvTransposedBlock2D, self).get_config()
        config.update(dict(
            filters=self.filters,
            strides=self.strides,
            dropout=self.dropout,
            final_shape=self.final_shape,
            activation=self.activation
        ))
        return config

    def build(self, input_shape):
        self.conv_transposed.build(input_shape)
        if self.final_shape is None:
            new_shape = self.conv_transposed.compute_output_shape(input_shape)
        else:
            new_shape = self.final_shape
        self.batch_norm.build(new_shape)
        self.activation_layer.build(new_shape)
        self.dropout_layer.build(new_shape)
        self.set_weights_variables(self.conv_transposed, self.batch_norm)
        super(ConvTransposedBlock2D, self).build(input_shape)

    def call(self, inputs):
        x = self.conv_transposed(inputs)
        if self.final_shape is not None:
            if x.shape[1] != self.final_shape[1]:       # step_length check
                x = x[:, :-1]
            if x.shape[2] != self.final_shape[2]:       # input_size check
                x = x[:, :, :-1]
        x = self.batch_norm(x)
        x = self.activation_layer(x)
        return self.dropout_layer(x)

    def compute_output_shape(self, input_shape):
        if self.final_shape is not None:
            return self.final_shape
        else:
            return self.conv_transposed.compute_output_shape(input_shape)


def new_shape_conv(input_shape: t.shape, strides: t.strides, filters: int) -> t.shape:
    """
    To handle with padding = 'same'
    number of dim = len(input_shapes) (without batch)

    :param input_shape: Tuple[int]:
        ⚠ batch dim is NOT in the tuple ⚠
    :param strides: Tuple[int]:
    :param filters: int:

    :return: Tuple[int]: The shape after a convolutional layer
    """
    new_shape = []
    for dim, stride in zip(input_shape[:-1], strides):
        new_shape.append(int(math.ceil(dim / stride)))
    new_shape.append(filters)
    return tuple(new_shape)


def new_shapes_conv(input_shape: t.shape, strides_list: t.List[t.strides], filters_list: t.List[int]
                    ) -> t.List[t.shape]:
    """
    Use to find the output shapes of several convolutional layers

    :param input_shape: Tuple[int]:
        ⚠ batch dim is NOT in shape ⚠
    :param strides_list: List[Tuple[int]]:
    :param filters_list: List[int]:

    :return:
    """
    new_shapes = [input_shape]
    for strides, filters in zip(strides_list, filters_list):
        new_shapes.append(new_shape_conv(new_shapes[-1], strides, filters))
    return new_shapes


def reverse_conv_param(original_dim: int, param_list: t.List[t.List[int]]) -> t.List[t.List[int]]:
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

    :param original_dim: int
    :param param_list: List[List[int]]: ex [[a, b], [c, d, e]] : the size of the convolutions

    :return: List[List[int]]: ex [[d, c, b], [a, original_dim]]: the size of the transposed convolutions
    """
    # --- (1) ---
    # --- (2) ---
    reversed_param_list_dims = [len(sublist) for sublist in param_list]
    reversed_param_list_temp = [size for sublist in param_list for size in sublist]  # Flatten the 2-level list
    # --- (3) ---
    reversed_param_list_temp = reversed_param_list_temp[::-1]  # Reversed
    reversed_param_list_dims = reversed_param_list_dims[::-1]
    # --- (4) ---
    reversed_param_list_temp = reversed_param_list_temp[1:] + [original_dim]  # Update shapes
    # --- (5) ---
    reversed_param_list = []  # Final reversed_param_list parameters
    offset = 0
    for sublist_size in reversed_param_list_dims:
        reversed_param_list.append(reversed_param_list_temp[offset: offset + sublist_size])
        offset += sublist_size
    return reversed_param_list
