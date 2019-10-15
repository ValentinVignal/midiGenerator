import tensorflow as tf

import src.global_variables as g
import src.mtypes as t

K = tf.keras.backend
layers = tf.keras.layers


class DenseBlock(layers.Layer):
    def __init__(self, units: int, dropout: float = g.dropout):
        """

        :param units: int
        :param dropout:
        """
        self.dense = layers.Dense(units)
        self.batch_norm = layers.BatchNormalization()
        self.leaky_relu = layers.LeakyReLU()
        self.dropout = layers.Dropout(dropout)
        super(DenseBlock, self).__init__()

    def build(self, input_shape):
        """

        :param input_shape: (?, input_size)
        :return:
        """
        self.dense.build(input_shape)
        new_shape = self.dense.compute_output_shape(input_shape)
        self.batch_norm.build(new_shape)
        self.leaky_relu.build(new_shape)
        self.dropout.build(new_shape)
        self._trainable_weights = self.dense.trainable_weights + self.batch_norm.trainable_weights
        self._non_trainable_weights = self.batch_norm.non_trainable_weights
        # TODO Verify there is no need to consider non_trainable_variable and trainable_variable
        super(DenseBlock, self).build(input_shape)

    def call(self, inputs):
        x = self.dense(inputs)
        x = self.batch_norm(x)
        x = self.leaky_relu(x)
        return self.dropout(x)

    def compute_output_shape(self, input_shape):
        return self.dense.compute_output_shape(input_shape)


class DenseCoder(layers.Layer):

    type_size_list = t.List[int]

    def __init__(self, size_list: type_size_list, dropout: float = g.dropout):
        """

        :param size_list: list<int>, (nb_blocks,)
        :param dropout: float
        """
        print('size list in dense', size_list)
        self.dense_blocks = []
        self.init_dense_blocks(size_list, dropout=dropout)
        super(DenseCoder, self).__init__()

    def init_dense_blocks(self, size_list: t.List, dropout: float = g.dropout):
        for size in size_list:
            self.dense_blocks.append(DenseBlock(size, dropout=dropout))

    def build(self, input_shape):
        """

        :param input_shape: (?, previous_size)
        :return:
        """
        new_shape = input_shape
        self._trainable_weights = []
        self._non_trainable_weights = []
        for dense in self.dense_blocks:
            dense.build(new_shape)
            new_shape = dense.compute_output_shape(new_shape)
            self._trainable_weights += dense.trainable_weights
            self._non_trainable_weights += dense.non_trainable_weights
            # TODO Verify there is no need to consider non_trainable_variable and trainable_variable
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


class DenseSameShape(layers.Layer):
    """
    Return a Dense layer which has the same shape as the inputs
    """
    def __init__(self, **kwargs):
        super(DenseSameShape, self).__init__()
        self.kwargs = kwargs
        self.dense = None
        self.already_built = False
        self.units = None

    def build(self, input_shape):
        """

        :param input_shape: (?, previous_size)
        :return:
        """
        # print('DenseSameShape build:  input shape', input_shape)
        if not self.already_built:
            self.units = input_shape[-1].value
            self.dense = layers.Dense(units=self.units, **self.kwargs)
            self.already_built = True
        self.dense.build(input_shape)
        self._trainable_weights = self.dense.trainable_weights
        self._non_trainable_weights = self.dense.non_trainable_weights
        # TODO Verify there is no need to consider non_trainable_variable and trainable_variable
        super(DenseSameShape, self).build(input_shape)

    def call(self, inputs):
        # print('DenseSameShape call: inputs', inputs, 'units', self.units)
        return self.dense(inputs)

    def compute_output_shape(self, input_shape):
        return self.dense.compute_output_shape(input_shape)



