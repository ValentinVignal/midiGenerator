import tensorflow as tf
import math

import src.NN.layers.dense as l_dense
import src.mtypes as t

K = tf.keras.backend
layers = tf.keras.layers


class Split(layers.Layer):
    """
    To split a tensor
    """

    type_num_or_size_to_split = t.Union[
        int,
        t.List[int]
    ]

    def __init__(self, num_or_size_to_split: type_num_or_size_to_split, axis: int = -1):
        """

        :param num_or_size_to_split: Union[int, List[int]]:
        :param axis: axis to split the tensor : int:
            ⚠ axis=0 correspond to the batch axis ⚠
        """
        super(Split, self).__init__()
        self.num_or_size_to_split = num_or_size_to_split
        self.axis = axis

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
        print('Split size_splits', size_splits)
        # print('Split', (*input_shape[:self.axis], 1, *input_shape[self.axis + 1:]))
        print('Split input_shape', input_shape)
        print('Split axis', self.axis)

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
                    shape.append(input_shape[ax].value)
            output_shape.append(tuple(shape))

        # output_shape = [(int(x) for x in (*input_shape[:self.axis], size_split, *input_shape[self.axis + 1:])) for size_split in
        #                 size_splits]
        print('Split compute_output_shape: output_shape', output_shape)
        return output_shape


class LastInstMono(layers.Layer):
    """
    Last layer for a model
    """

    def __init__(self, softmax_axis: int):
        """

        :param softmax_axis: int:
        """
        super(LastInstMono, self).__init__()
        self.softmax_axis = softmax_axis

        self.already_built = False

        self.flatten = layers.Flatten()
        self.dense = l_dense.DenseSameShape()
        self.reshape = None
        self.softmax = layers.Softmax(axis=softmax_axis)

    def build(self, input_shape):
        print('LastInstMono input_shape', input_shape)
        self.flatten.build(input_shape)
        new_shape = self.flatten.compute_output_shape(input_shape)
        print('LastInstMono new shape after flatten', new_shape)
        self.dense.build(new_shape)
        print('LastInstMono new shape after dense ', new_shape)
        print('LastInstMono input shape in last Inst mono ', input_shape)
        new_shape = self.dense.compute_output_shape(new_shape)
        if not self.already_built:
            self.reshape = layers.Reshape(input_shape[1:])  # Don't take the batch shape
            self.already_built = True
        self.reshape.build(new_shape)
        new_shape = self.reshape.compute_output_shape(new_shape)
        self.softmax.build(new_shape)

        self._trainable_weights = self.dense.trainable_weights
        self._non_trainable_weights = self.dense.non_trainable_weights
        # TODO Verify there is no need to consider non_trainable_variable and trainable_variable
        super(LastInstMono, self).build(input_shape)

    def call(self, inputs):
        print('LastInstMono call: inputs', inputs.shape)
        x = self.flatten(inputs)
        x = self.dense(x)
        x = self.reshape(x)
        x = self.softmax(x)
        return x

    def compute_output_shape(self, input_shape):
        return input_shape


class LastMono(layers.Layer):
    def __init__(self, softmax_axis: int, names: t.Optional[str] = None):
        """

        :param softmax_axis:
        :param names:
        """
        super(LastMono, self).__init__()
        self.sofmax_axis = softmax_axis
        self.names = names

        self.split = None
        self.last_inst_mono_list = []
        self.already_build = False
        self.nb_instruments = None

    def build(self, input_shape):
        if not self.already_build:
            self.nb_instruments = int(input_shape[-1])
            self.split = Split(num_or_size_to_split=self.nb_instruments, axis=-1)
            for inst in range(self.nb_instruments):
                self.last_inst_mono_list.append(LastInstMono(softmax_axis=self.sofmax_axis))
            self.already_build = True
        self.split.build(input_shape)
        print('LastMono input_shape', input_shape)
        new_shapes = self.split.compute_output_shape(input_shape)
        print('LastMono new_shape after split', new_shapes)
        self._trainable_weights = []
        self._non_trainable_weights = []
        for inst in range(self.nb_instruments):
            self.last_inst_mono_list[inst].build(new_shapes[inst])
            self._trainable_weights += self.last_inst_mono_list[inst].trainable_weights
            self._non_trainable_weights += self.last_inst_mono_list[inst].non_trainable_weights
            # TODO Verify there is no need to consider non_trainable_variable and trainable_variable
        super(LastMono, self).build(input_shape)

    def call(self, inputs):
        x = self.split(inputs)
        x = [self.last_inst_mono_list[inst](x[inst]) for inst in range(self.nb_instruments)]
        return x

    def compute_output_shape(self, input_shape):
        return self.split.compute_output_shape(input_shape)
