import tensorflow as tf
import src.global_variables as g

K = tf.keras.backend
layers = tf.keras.layers


class DenseBlock(layers.Layer):
    def __init__(self, units, dropout=g.dropout):
        self.dense = layers.Dense(units)
        self.batch_norm = layers.BatchNormalization()
        self.leaky_relu = layers.LeakyReLU()
        self.dropout = layers.Dropout(dropout)
        super(DenseBlock, self).__init__()

    def build(self, input_shape):
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
    def __init__(self, size_list, dropout=g.dropout):
        self.dense_blocks = []
        self.init_dense_blocks(size_list, dropout=dropout)
        super(DenseCoder, self).__init__()

    def init_dense_blocks(self, size_list, dropout=g.dropout):
        for size in size_list:
            self.dense_blocks.append(DenseBlock(size, dropout=dropout))

    def build(self, input_shape):
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
    def __init__(self, **kwargs):
        super(DenseSameShape, self).__init__()
        self.kwargs = kwargs
        self.dense = None
        self.already_built = False

    def build(self, input_shape):
        if not self.already_built:
            self.dense = layers.Dense(units=input_shape[-1], **self.kwargs)
            self.already_built = True
        self.dense.build()
        self._trainable_weights = self.dense.trainable_weights
        self._non_trainable_weights = self.dense.non_trainable_weights
        # TODO Verify there is no need to consider non_trainable_variable and trainable_variable
        super(DenseSameShape, self).build(input_shape)

    def call(self, inputs):
        return self.dense(inputs)

    def compute_output_shape(self, input_shape):
        return self.dense.compute_output_shape(input_shape)



