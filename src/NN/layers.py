from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.keras.engine.base_layer import Layer, InputSpec
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras import constraints

K = tf.keras.backend
layers = tf.keras.layers


class BatchNormalization(layers.Layer):

    def __init__(self, axis=-1, momentum=0.999, epsilon=1e-3, **kwargs):
        super(BatchNormalization, self).__init__(**kwargs)
        self.axis = axis
        self.momentum = momentum
        self.epsilon = epsilon

        self.shape = None

        self._axis = None
        self.scale = None
        self.beta = None
        self.pop_mean = None
        self.pop_var = None

    def build(self, inputs_shape):
        # ----- Shape -----
        if type(self.axis) is int:
            shape = (inputs_shape[self.axis], )
            self._axis = [i for i in range(len(inputs_shape))]
            self._axis = self._axis[:self.axis] + self._axis[self.axis+1:]
        elif type(self.axis) is list or type(self.axis) is tuple:
            shape = []
            axis_abs = []
            for i in range(len(self.axis)):
                shape.append(inputs_shape[self.axis[i]])
                if self.axis[i] >= 0:
                    axis_abs.append(self.axis[i])
                else:
                    axis_abs.append(len(inputs_shape) - self.axis[i])
            shape = tuple(shape)
            self._axis = []
            for i in range(len(inputs_shape)):
                if i not in axis_abs:
                    self._axis.append(i)
        else:
            raise TypeError('{0}: axis should be an int, list or tuple. Not a {1}'.format(self.name, type(self.axis)))
        self.shape = shape
        # ----- _axis -----

        self.scale = self.add_weight(name='batch_norm_scale',
                                     shape=shape,
                                     initializer=tf.keras.initializers.ones)
        self.beta = self.add_weight(name='batch_norm_beta',
                                    shape=shape,
                                    initializer=tf.keras.initializers.zeros)
        self.pop_mean = self.add_weight(name='batch_norm_pop_mean',
                                        shape=shape,
                                        initializer=tf.keras.initializers.zeros,
                                        trainable=False)
        self.pop_var = self.add_weight(name='batch_norm_pop_var',
                                       shape=shape,
                                       initializer=tf.keras.initializers.ones,
                                       trainable=False)

    def call(self, inputs, training=None):

        def batch_norm_train():
            batch_mean, batch_var = tf.nn.moments(inputs, self._axis)
            train_mean = tf.assign(self.pop_mean,
                                   self.pop_mean * self.momentum + batch_mean * (1 - self.momentum))
            train_var = tf.assign(self.pop_var,
                                  self.pop_var * self.momentum + batch_var * (1 - self.momentum))
            with tf.control_dependencies([train_mean, train_var]):
                return tf.nn.batch_normalization(inputs,
                    batch_mean, batch_var, self.scale, self.beta, self.epsilon)

        def batch_norm_no_train():
            return tf.nn.batch_normalization(inputs,
                                         self.pop_mean, self.pop_var, self.scale, self.beta, self.epsilon)

        return K.in_train_phase(batch_norm_train, batch_norm_no_train, training=training)

    def compute_output_shape(self, input_shape):
        return input_shape

    # TODO(not implemented): code get_config method

