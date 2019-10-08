import tensorflow as tf
import src.global_variables as g

K = tf.keras.backend
layers = tf.keras.layers


class ConvBlock3D(layers.Layer):
    def __init__(self, filters, strides=(1, 1, 1), dropout=g.dropout):
        self.strides = strides
        self.conv = layers.Conv3D(filters=filters,
                                  kernel_size=(1, 5, 5),
                                  strides=strides,
                                  padding='same')
        self.batch_norm = layers.BatchNormalization()
        self.leaky_relu = layers.LeakyReLU()
        self.dropout = layers.Dropout(dropout)
        super(ConvBlock3D, self).__init__()

    def build(self, input_shape):
        self.conv.build(input_shape)
        new_shape = self.conv.compute_output_shape(input_shape)
        self.batch_norm.build(new_shape)
        self.leaky_relu.build(new_shape)
        self.dropout.build(new_shape)
        self._trainable_weights = self.conv.trainable_weights + self.batch_norm.trainable_weights
        self._non_trainable_weights = self.batch_norm.non_trainable_weights
        # TODO Verify there is no need to consider non_trainable_variable and trainable_variable
        super(ConvBlock3D, self).build(input_shape)

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.batch_norm(x)
        x = self.leaky_relu(x)
        return self.dropout(x)

    def compute_output_shape(self, input_shape):
        return self.conv.compute_output_shape(input_shape)


class ConvTransposedBlock3D(layers.Layer):
    def __init__(self, filters, strides=(1, 1, 1), dropout=g.dropout, final_shape=None):
        self.conv_transposed = layers.Conv3DTranspose(filters=filters,
                                                      kernel_size=(1, 5, 5),
                                                      padding='same',
                                                      strides=strides)
        self.batch_norm = layers.BatchNormalization()
        self.leaky_relu = layers.LeakyReLU()
        self.dropout = layers.Dropout(dropout)
        super(ConvTransposedBlock3D, self).__init__()

        self.final_shape = final_shape

    def build(self, input_shape):
        self.conv_transposed.build(input_shape)
        if self.final_shape is None:
            new_shape = self.conv_transposed.compute_output_shape(input_shape)
        else:
            new_shape = self.final_shape
        self.batch_norm.build(new_shape)
        self.leaky_relu.build(new_shape)
        self.dropout.build(new_shape)
        self._trainable_weights = self.conv_transposed.trainable_weights + self.batch_norm.trainable_weights
        self._non_trainable_weights = self.batch_norm.non_trainable_weights
        # TODO Verify there is no need to consider non_trainable_variable and trainable_variable
        super(ConvTransposedBlock3D, self).build(input_shape)

    def call(self, inputs):
        x = self.conv_transposed(inputs)
        if self.final_shape is not None:
            if x.shape[3] != self.final_shape[3]:
                x = x[:, :, :, :-1]
            if x.shape[2] != self.final_shape[2]:
                x = x[:, :, :-1]
        x = self.batch_norm(x)
        x = self.leaky_relu(x)
        return self.dropout(x)

    def compute_output_shape(self, input_shape):
        return self.conv_transposed.compute_output_shape(input_shape)



