import tensorflow as tf
from .KerasLayer import KerasLayer
import warnings

layers = tf.keras.layers
K = tf.keras.backend
math = tf.math


class ProductOfExpert(KerasLayer):
    """

    """

    def __init__(self, axis=0, eps=1e-8):
        """

        :param axis: (not batch in it)
        """
        super(ProductOfExpert, self).__init__()

        self.axis = axis
        self.axis_with_batch = axis + 1
        self.eps = eps

    def build(self, input_shape):
        super(ProductOfExpert, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        """

        :param input_shape: List(2)[
                            (batch, modalities, size)
                        ]
        :return: List(2)[
                    (batch, size)
                ]
        """
        input_shape = input_shape[0]
        return [(*input_shape[:self.axis_with_batch], *input_shape[self.axis_with_batch+1:]) for i in range(2)]

    def call(self, inputs):
        """

        :param inputs: List(2)[(batch, modalities, size)]
            ([means, std])
        :return:
        """
        var = inputs[1] ** 2 + self.eps
        T = 1 / var
        T = tf.where(math.is_nan(T), tf.zeros_like(T), T)
        mean = inputs[0]
        mean = tf.where(math.is_nan(mean), tf.zeros_like(mean), mean)
        product_mean = math.reduce_sum(mean * T, axis=self.axis_with_batch) / (
                    math.reduce_sum(T, axis=self.axis_with_batch) + self.eps)
        product_std = math.sqrt(1 / math.reduce_sum(T, axis=self.axis_with_batch))
        return [product_mean, product_std]


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
