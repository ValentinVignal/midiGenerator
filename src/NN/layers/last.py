import tensorflow as tf
import math

from . import dense as l_dense
import src.mtypes as t
from.KerasLayer import KerasLayer
from . import shapes as shapes

K = tf.keras.backend
layers = tf.keras.layers
math = tf.math


class Split(KerasLayer):
    """
    To split a tensor
    """

    type_num_or_size_to_split = t.Union[
        int,
        t.List[int]
    ]

    def __init__(self, num_or_size_to_split: type_num_or_size_to_split, axis: int = -1, *args, **kwargs):
        """

        :param num_or_size_to_split: Union[int, List[int]]:
        :param axis: axis to split the tensor : int:
            ⚠ axis=0 correspond to the batch axis ⚠
        """
        super(Split, self).__init__(*args, **kwargs)
        # ---------- Raw parameters ----------
        self.num_or_size_to_split = num_or_size_to_split
        self.axis = axis

    def get_config(self):
        config = super(Split, self).get_config()
        config.update(dict(
            num_or_size_to_split=self.num_or_size_to_split,
            axis=self.axis
        ))
        return config

    def build(self, input_shape):
        super(Split, self).build(input_shape)

    def call(self, inputs):
        output = list(tf.split(inputs,
                               num_or_size_splits=self.num_or_size_to_split,
                               axis=self.axis))
        return output

    def compute_output_shape(self, input_shape):
        if isinstance(self.num_or_size_to_split, list):
            size_splits = self.num_or_size_to_split
        else:
            size_splits = [math.floor(int(input_shape[self.axis]) / self.num_or_size_to_split) for i in
                           range(self.num_or_size_to_split)]
        if self.axis > 0:
            normalized_axis = self.axis
        else:
            normalized_axis = len(input_shape) + self.axis

        output_shape = []
        for size_split in size_splits:
            shape = []
            for ax in range(len(input_shape)):
                if ax == normalized_axis:
                    shape.append(size_split)
                else:
                    shape.append(shapes.get_shape(input_shape, ax))
            output_shape.append(tuple(shape))

        return output_shape


class LastInstMono(KerasLayer):
    """
    Last layer for a model
    """

    def __init__(self, softmax_axis: int, *args, **kwargs):
        """
        :param softmax_axis: int:
        """
        super(LastInstMono, self).__init__(*args, **kwargs)
        # --------- Raw parameters ----------
        self.softmax_axis = softmax_axis

        self.already_built = False

        self.flatten = layers.Flatten()
        self.dense = l_dense.DenseSameShape()
        self.reshape = None
        self.softmax = layers.Softmax(axis=softmax_axis)

    def get_config(self):
        config = super(LastInstMono, self).get_config()
        config.update(dict(
            softmax_axis=self.softmax_axis
        ))
        return config

    def build(self, input_shape):
        self.flatten.build(input_shape)
        new_shape = self.flatten.compute_output_shape(input_shape)
        self.dense.build(new_shape)
        new_shape = self.dense.compute_output_shape(new_shape)
        if not self.already_built:
            self.reshape = layers.Reshape(input_shape[1:])
            self.already_built = True
        self.reshape.build(new_shape)
        new_shape = self.reshape.compute_output_shape(new_shape)
        self.softmax.build(new_shape)

        self.set_weights_variables(self.dense)
        super(LastInstMono, self).build(input_shape)

    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.dense(x)
        x = self.reshape(x)
        x = self.softmax(x)
        return x

    def compute_output_shape(self, input_shape):
        return input_shape


class LastInstMonoBinary(KerasLayer):
    """
    Last layer for a model
    """

    def __init__(self, softmax_axis: int, *args, **kwargs):
        """

        :param softmax_axis: int:
        """
        super(LastInstMonoBinary, self).__init__(*args, **kwargs)
        # --------- Raw parameters ----------
        self.softmax_axis = softmax_axis

        self.already_built = False

        self.flatten = layers.Flatten()
        self.dense = l_dense.DenseSameShape()
        self.reshape = None
        self.softmax = layers.Softmax(axis=softmax_axis)

    def get_config(self):
        config = super(LastInstMonoBinary, self).get_config()
        config.update(dict(
            softmax_axis=self.softmax_axis
        ))
        return config

    def build(self, input_shape):
        """

        :param input_shape: (batch, step_length, size, channels=1)
        :return:
        """
        self.flatten.build(input_shape)
        new_shape = self.flatten.compute_output_shape(input_shape)
        self.dense.build(new_shape)
        new_shape = self.dense.compute_output_shape(new_shape)
        if not self.already_built:
            self.reshape = layers.Reshape(input_shape[1:])
            self.already_built = True
        self.reshape.build(new_shape)
        new_shape = self.reshape.compute_output_shape(new_shape)
        new_shape = (*new_shape[:2], new_shape[2] - 1, *new_shape[3:])
        # Removing activation for softmax function
        self.softmax.build(new_shape)

        self.set_weights_variables(self.dense)
        super(LastInstMonoBinary, self).build(input_shape)

    def call(self, inputs):
        """

        :param inputs: (batch, step_length, size, channels=1)
        :return:
        """
        x = self.flatten(inputs)
        x = self.dense(x)
        x = self.reshape(x)     # (batch, step_length, size, channels)
        x_c = self.softmax(x[:, :, :-1])      # (batch, step_length, size-1, channels=1)
        x_b = math.sigmoid(x[:, :, -1:])    # (batch, step_length, 1, channels=1)
        x = tf.concat([x_c, x_b], axis=2)
        return x

    def compute_output_shape(self, input_shape):
        return input_shape


class LastMono(KerasLayer):
    def __init__(self, softmax_axis: int, names: t.Optional[str] = None, *args,  **kwargs):
        """

        :param softmax_axis:
        :param names:
        """
        super(LastMono, self).__init__(*args, **kwargs)
        # ---------- Raw parameters ----------
        self.softmax_axis = softmax_axis
        self.names = 'last_mono' if names is None else names

        self.split = None
        self.last_inst_mono_list = []
        self.already_build = False
        self.nb_instruments = None

    def get_config(self):
        config = super(LastMono, self).get_config()
        config.update(dict(
            softmax_axis=self.softmax_axis,
            names=self.names
        ))
        return config

    def build(self, input_shape):
        if not self.already_build:
            self.nb_instruments = int(input_shape[-1])
            self.split = Split(num_or_size_to_split=self.nb_instruments, axis=-1)
            for inst in range(self.nb_instruments):
                self.last_inst_mono_list.append(
                    LastInstMonoBinary(softmax_axis=self.softmax_axis))
            self.already_build = True
        self.split.build(input_shape)
        new_shapes = self.split.compute_output_shape(input_shape)
        self.reset_weights_variables()
        for inst in range(self.nb_instruments):
            self.last_inst_mono_list[inst].build(new_shapes[inst])
            self.add_weights_variables(self.last_inst_mono_list[inst])
        super(LastMono, self).build(input_shape)

    def call(self, inputs):
        x = self.split(inputs)
        x = [self.last_inst_mono_list[inst](x[inst]) for inst in range(self.nb_instruments)]
        return x

    def compute_output_shape(self, input_shape):
        return self.split.compute_output_shape(input_shape)


