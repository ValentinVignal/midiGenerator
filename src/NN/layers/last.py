import tensorflow as tf

import src.NN.layers.dense as l_dense

K = tf.keras.backend
layers = tf.keras.layers


class LastInstMono(layers.Layer):
    def __init__(self, instrument, softmax_axis):
        super(LastInstMono, self).__init__()
        self.instrument = instrument
        self.softmax_axis = softmax_axis

        self.already_built = False

        self.mlambda = layers.Lambda(self.choose_instrument)
        self.flatten = layers.Flatten()
        self.dense = l_dense.DenseSameShape()
        self.reshape = None
        self.softmax = layers.Softmax(axis=softmax_axis)

    def choose_instrument(self, x):
        return x[..., self.instrument]

    def build(self, input_shape):
        self.mlambda.build(input_shape)
        new_shape = self.mlambda.compute_output_shape(input_shape)
        self.flatten.build(new_shape)
        new_shape = self.flatten.compute_output_shape(new_shape)
        self.dense.build(new_shape)
        new_shape = self.dense.compute_output_shape(new_shape)
        self.reshape.build(new_shape)
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
        x = self.mlambda(inputs)
        x = self.flatten(x)
        x = self.dense(x)
        x = self.reshape(x)
        x = self.softmax(x)
        return x

    def compute_output_shape(self, input_shape):
        return input_shape


class LastMono(layers.Layer):
    def __init__(self, softmax_axis):
        super(LastMono, self).__init__()
        self.sofmax_axis = softmax_axis

        self.last_inst_mono_list = []
        self.already_build = False
        self.nb_instruments = None

    def build(self, input_shape):
        if not self.already_build:
            self.nb_instruments = input_shape[-1]
            for inst in range(self.nb_instruments):
                self.last_inst_mono_list.append(LastInstMono(instrument=inst, softmax_axis=self.sofmax_axis))
            self.already_build = True
        self._trainable_weights = []
        self._non_trainable_weights = []
        for inst in range(self.nb_instruments):
            self.last_inst_mono_list[inst].build(input_shape)
            self._trainable_weights += self.last_inst_mono_list[inst].trainable_weights
            self._non_trainable_weights += self.last_inst_mono_list[inst].non_trainable_weights
            # TODO Verify there is no need to consider non_trainable_variable and trainable_variable
        super(LastMono, self).build(input_shape)

    def call(self, inputs):
        x = [self.last_inst_mono_list[inst](inputs) for inst in range(self.nb_instruments)]
        return x

    def compute_output_shape(self, input_shape):
        output_shape = [self.last_inst_mono_list[inst].compute_output_shape(input_shape) for inst in
                        range(self.nb_instruments)]
        return output_shape
