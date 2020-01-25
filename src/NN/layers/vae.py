import tensorflow as tf
from .KerasLayer import KerasLayer

layers = tf.keras.layers
K = tf.keras.backend
math = tf.math

from src.NN import Loss


class ProductOfExpert(KerasLayer):
    """

    """

    def __init__(self, axis=0, eps=1e-8, *args, **kwargs):
        """

        :param axis: (not batch in it)
        """
        super(ProductOfExpert, self).__init__(*args, **kwargs)
        # ---------- Raw parameters ----------
        self.axis = axis
        self.eps = eps

        self.axis_with_batch = axis + 1

    def get_config(self):
        config = super(ProductOfExpert, self).get_config()
        config.update(dict(
            axis=self.axis,
            eps=self.eps
        ))
        return config

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
        product_std = math.sqrt(1 / (math.reduce_sum(T, axis=self.axis_with_batch) + self.eps))
        return [product_mean, product_std]


class ProductOfExpertMask(KerasLayer):
    """

    """

    def __init__(self, axis=0, eps=1e-8, *args, **kwargs):
        """

        :param axis: (not batch in it) : Axis of the modalities
        """
        super(ProductOfExpertMask, self).__init__(*args, **kwargs)
        # ---------- Raw parameters ----------
        self.axis = axis
        self.eps = eps

        self.axis_with_batch = axis + 1

    def get_config(self):
        config = super(ProductOfExpertMask, self).get_config()
        config.update(dict(
            axis=self.axis,
            eps=self.eps
        ))
        return config

    def build(self, input_shape):
        super(ProductOfExpertMask, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        """

        :param input_shape: List(2)[
                            (batch, modalities, nb_steps?, size)
                        ] + List[1](batch, modalities, nb_steps?)
        :return: List(2)[
                    (batch, nb_steps?, size)
                ]
        """
        input_shape = input_shape[0]
        return [(*input_shape[:self.axis_with_batch], *input_shape[self.axis_with_batch+1:]) for i in range(2)]

    def call(self, inputs):
        """

        :param inputs: List(2)[(batch, modalities, nb_steps?, size)] + List(1)[(batch, modalities, nb_steps?)]
            ([means, std])
        :return:
        """
        mask = tf.expand_dims(inputs[2], axis=-1)        # (batch, modalities, nb_step?, 1)
        var = inputs[1] ** 2 + self.eps
        T = 1 / var
        # T = tf.where(math.is_nan(T), tf.zeros_like(T), T)
        T = T * mask
        mean = inputs[0]
        # mean = tf.where(math.is_nan(mean), tf.zeros_like(mean), mean)
        mean = mean * mask
        product_mean = math.reduce_sum(mean * T, axis=self.axis_with_batch) / (
                math.reduce_sum(T, axis=self.axis_with_batch) + self.eps)
        product_std = math.sqrt(1 / (math.reduce_sum(T, axis=self.axis_with_batch) + self.eps))
        return [product_mean, product_std]


class NaNToZeros(KerasLayer):
    """

    """

    def __init__(self, *args, **kwargs):
        super(NaNToZeros, self).__init__(*args, **kwargs)

    def get_config(self):
        config = super(NaNToZeros, self).get_config()
        return config

    def build(self, input_shape):
        """

        :return:
        """
        super(NaNToZeros, self).build(input_shape)

    def call(self, inputs):
        return tf.where(tf.math.is_nan(inputs), tf.zeros_like(inputs), inputs)

    def compute_output_shape(self, input_shape):
        return input_shape


class SampleGaussian(KerasLayer):
    """
    Samples a Gaussian N(0, 1)

    """
    def call(self, inputs):
        z_mean, z_std = inputs
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        sample = z_mean + tf.exp(z_std) * epsilon
        return sample

    def compute_output_shape(self, input_shape):
        return input_shape[0]


class KLD(KerasLayer):
    """

    """
    def __init__(self, weight=1, sum_axis=None, *args, **kwargs):
        """

        :param weight:
        :param sum_axis: To sum through time. If is None, no summation is done
        :param kwargs:
        """
        super(KLD, self).__init__(*args, **kwargs)
        # ---------- Raw parameters ----------
        self.weigth = weight
        self.sum_axis = sum_axis

        self.sum_axis_with_batch = self.compute_sum_axis_with_batch(sum_axis)

    @staticmethod
    def compute_sum_axis_with_batch(sum_axis):
        if sum_axis is None:
            return sum_axis
        else:
            if isinstance(sum_axis, int):
                sum_axis = [sum_axis]
            sum_axis_with_batch = []
            for axis in sum_axis:
                if axis >= 0:
                    sum_axis_with_batch.append(axis + 1)
                else:
                    sum_axis_with_batch.append(axis)
            return sum_axis_with_batch

    def get_config(self):
        config = super(KLD, self).get_config()
        config.update(dict(
            weight=self.weigth,
            sum_axis=self.sum_axis
        ))
        return config

    def call(self, inputs):
        mean, std = inputs
        return self.weigth * Loss.cost.kld(mean, std, sum_axis=self.sum_axis_with_batch)

    def compute_output_shape(self, input_shape):
        return 1,




