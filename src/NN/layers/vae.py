import tensorflow as tf
from .layers import KerasLayer
import warnings

layers = tf.keras.layers
K = tf.keras.backend


class ProductOfExpert(KerasLayer):
    """

    """
    def __init__(self, axis=0):
        """

        :param axis: (not batch in it)
        """
        super(ProductOfExpert, self).__init__()

        self.step_axis = axis

    def build(self, input_shape):
        print('input_shape', isinstance(input_shape, list), len(input_shape), input_shape[0])


class NaNToZeros(KerasLayer):
    """

    """
    def __init__(self):
        super(NaNToZeros, self).__init__()

    def build(self, input_shape):
        """

        :return:
        """
        super(NaNToZeros, self).build(input_shape)

    def call(self, inputs):
        return tf.where(tf.math.is_nan(inputs), tf.zeros_like(inputs), inputs)

    def compute_output_shape(self, input_shape):
        return input_shape



