import tensorflow as tf

import src.global_variables as g
import src.mtypes as t
from .KerasLayer import KerasLayer
from . import shapes as shapes

K = tf.keras.backend
layers = tf.keras.layers


class DenseBlock(KerasLayer):
    def __init__(self, units: int, dropout: float = g.dropout, *args, **kwargs):
        """

        :param units: int
        :param dropout:
        """
        super(DenseBlock, self).__init__(*args, **kwargs)
        # ---------- Raw parameters ----------
        self.units = units,
        self.dropout = dropout

        self.dense = layers.Dense(units)
        self.batch_norm = layers.BatchNormalization()
        self.leaky_relu = layers.LeakyReLU()
        self.dropout_layer = layers.Dropout(dropout)

    def get_config(self):
        config = super(DenseBlock, self).get_config()
        config.update(dict(
            units=self.units,
            dropout=self.dropout
        ))

    def build(self, input_shape):
        """

        :param input_shape: (?, input_size)
        :return:
        """
        self.dense.build(input_shape)
        new_shape = self.dense.compute_output_shape(input_shape)
        self.batch_norm.build(new_shape)
        self.leaky_relu.build(new_shape)
        self.dropout_layer.build(new_shape)
        self.set_weights_variables(self.dense, self.batch_norm)
        super(DenseBlock, self).build(input_shape)

    def call(self, inputs):
        x = self.dense(inputs)
        x = self.batch_norm(x)
        x = self.leaky_relu(x)
        return self.dropout_layer(x)

    def compute_output_shape(self, input_shape):
        return self.dense.compute_output_shape(input_shape)


class DenseCoder(KerasLayer):

    type_size_list = t.List[int]

    def __init__(self, size_list: type_size_list, dropout: float = g.dropout, *args, **kwargs):
        """

        :param size_list: list<int>, (nb_blocks,)
        :param dropout: float
        """
        super(DenseCoder, self).__init__(*args, **kwargs)
        # ---------- Raw parameters ----------
        self.size_list = size_list
        self.dropout =dropout

        self.dense_blocks = []
        self.init_dense_blocks(size_list, dropout=dropout)

    def init_dense_blocks(self, size_list: t.List, dropout: float = g.dropout):
        for size in size_list:
            self.dense_blocks.append(DenseBlock(size, dropout=dropout))

    def get_config(self):
        config = super(DenseCoder, self).get_config()
        config.update(dict(
            size_list=self.size_list,
            dropout=self.dropout
        ))
        return config

    def build(self, input_shape):
        """

        :param input_shape: (?, previous_size)
        :return:
        """
        new_shape = input_shape
        self.reset_weights_variables()
        for dense in self.dense_blocks:
            dense.build(new_shape)
            new_shape = dense.compute_output_shape(new_shape)
            self.add_weights_variables(dense)
        super(DenseCoder, self).build(input_shape)

    def call(self, inputs):
        x = inputs
        for dense in self.dense_blocks:
            x = dense(x)
        return x

    def compute_output_shape(self, input_shape):
        new_shape = input_shape
        for dense in self.dense_blocks:
            new_shape = dense.compute_output_shape(new_shape)
        return new_shape


class DenseSameShape(KerasLayer):
    """
    Return a Dense layer which has the same shape as the inputs
    """
    def __init__(self, *args, **kwargs):
        super(DenseSameShape, self).__init__(*args, **kwargs)
        self.kwargs = kwargs
        self.dense = None
        self.already_built = False
        self.units = None

    def get_config(self):
        config = super(DenseSameShape, self).get_config()
        config.update(self.kwargs)
        return config

    def build(self, input_shape):
        """

        :param input_shape: (?, previous_size)
        :return:
        """
        if not self.already_built:
            self.units = shapes.get_shape(input_shape, -1)
            self.dense = layers.Dense(units=self.units, **self.kwargs)
            self.already_built = True
        self.dense.build(input_shape)
        self.set_weights_variables(self.dense)
        super(DenseSameShape, self).build(input_shape)

    def call(self, inputs):
        # print('DenseSameShape call: inputs', inputs, 'units', self.units)
        return self.dense(inputs)

    def compute_output_shape(self, input_shape):
        return self.dense.compute_output_shape(input_shape)


class NDense(KerasLayer):
    """
    Return a list of N tensor from Denses layers
    """
    def __init__(self, units, n=2, *args, **kwargs):
        super(NDense, self).__init__(*args, **kwargs)
        # ---------- Raw parameters ----------
        self.units = units
        self.n = n

        self.units, self.n = self.verify_attributs(units, n)
        self.denses = [layers.Dense(self.units[u]) for u in units]

    @staticmethod
    def verify_attributs(units, n=2):
        """

        :param units:
        :param n:
        :return:
        """
        if isinstance(units, list):
            n = len(units)
            return units, n
        elif isinstance(units, int):
            units = [units for i in range(n)]
            return units, n

    def get_config(self):
        config = super(NDense, self).get_config()
        config.update(dict(
            units=self.units,
            n=self.n
        ))
        return config

    def build(self, input_shape):
        for dense in self.denses:
            dense.build(input_shape)
        super(NDense, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return [dense.compute_output_shape(input_shape) for dense in self.denses]

    def call(self, inputs):
        return [dense(inputs) for dense in self.denses]


class DenseForMean(KerasLayer):
    """
       Used to compute the mean of something using a Dense Layer
    """
    def __init__(self, units, *args, **kwargs):
        super(DenseForMean, self).__init__(*args, **kwargs)
        # ---------- Raw parameters ----------
        self.units = units

        self.dense = layers.Dense(units)

    def get_config(self):
        config = super(DenseForMean, self).get_config()
        config.update(dict(
            units=self.units
        ))
        return config

    def build(self, input_shape):
        self.dense.build(input_shape)
        super(DenseForMean, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return self.compute_output_shape(input_shape)

    def call(self, inputs):
        return self.dense(inputs)


class DenseForSTD(KerasLayer):
    """
       Used to compute the mean of something using a Dense Layer
    """

    def __init__(self, units, *args, **kwargs):
        super(DenseForSTD, self).__init__(*args, **kwargs)
        # ---------- Raw parameters ----------
        self.units = units

        self.dense = layers.Dense(units)

    def get_config(self):
        config = super(DenseForSTD, self).get_config()
        config.update(dict(
            units=self.units
        ))
        return config

    def build(self, input_shape):
        self.dense.build(input_shape)
        super(DenseForSTD, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return self.compute_output_shape(input_shape)

    def call(self, inputs):
        x = self.dense(inputs)
        x = tf.keras.activations.softplus(x)
        return x
