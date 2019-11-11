import tensorflow as tf
import warnings

from .layers import KerasLayer


def get_shape(t, ax):
    """

    :param t:
    :param ax:
    :return:
    """
    if t[ax] is None or isinstance(t[ax], int):
        return t[ax]
    else:
        return t[ax].value


class SwitchListAxis(KerasLayer):
    """

    """

    def __init__(self, axis=0):
        """

        :param axis: Batch axis is not included
        """
        super(SwitchListAxis, self).__init__()
        self.axis =axis
        self.perms = None

    def build(self, input_shape):
        self.verify_shapes(input_shape)
        self.get_perms(input_shape)
        super(SwitchListAxis, self).build(input_shape)

    def get_perms(self, input_shape):
        """

        :param input_shape:
        :return:
        """
        nb_dim = len(input_shape[0]) + 1        # + new axis from stack
        perms = list(range(nb_dim))
        perms[0] = self.axis + 2        # batch + new axis from stack
        perms[self.axis+2] = 0
        self.perms = perms

    @staticmethod
    def verify_shapes(input_shape):
        for i in range(1, len(input_shape)):
            if input_shape[i - 1] != input_shape[i]:
                warnings.warn(
                    f"All the shapes should be have the same dimension: shape {i - 1}: {input_shape[i - 1]} != {i}: {input_shape[i]}"
                )

    def compute_output_shape(self, input_shape):
        len_list = len(input_shape)
        axis_with_batch = self.axis + 1
        new_len_list = input_shape[0][axis_with_batch]

        return [(*input_shape[0][:axis_with_batch], len_list, *input_shape[0][axis_with_batch+1:]) for i in range(new_len_list)]

    def call(self, inputs):
        """

        :param inputs: List[tensor]
        :return:
        """
        inputs = tf.stack(inputs, axis=0)       # Tensor
        outputs = tf.transpose(inputs, perm=self.perms)     # Tensor
        outputs = tf.unstack(outputs)       # List[tensor]
        return outputs


class Stack(KerasLayer):
    def __init__(self, axis=0):
        """

        :param axis: No batch in axis
        """
        super(Stack, self).__init__()
        self.axis = axis
        self.axis_with_batch = axis + 1 if axis >= 0 else axis

    def build(self, input_shape):
        super(Stack, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        length = len(input_shape)
        input_shape = input_shape[0]
        return (*input_shape[:self.axis_with_batch], length, *input_shape[self.axis_with_batch:])

    def call(self, inputs):
        return tf.stack(inputs, axis=self.axis_with_batch)


