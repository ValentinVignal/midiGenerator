from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.keras.engine.base_layer import Layer, InputSpec
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras import constraints

import tensorflow.python.keras.backend as K

'''
class BatchNormalization(Layer):
    """Batch normalization layer (Ioffe and Szegedy, 2014).
    Normalize the activations of the previous layer at each batch,
    i.e. applies a transformation that maintains the mean activation
    close to 0 and the activation standard deviation close to 1.
    # Arguments
        axis: Integer, the axis that should be normalized
            (typically the features axis).
            For instance, after a `Conv2D` layer with
            `data_format="channels_first"`,
            set `axis=1` in `BatchNormalization`.
        momentum: Momentum for the moving mean and the moving variance.
        epsilon: Small float added to variance to avoid dividing by zero.
        center: If True, add offset of `beta` to normalized tensor.
            If False, `beta` is ignored.
        scale: If True, multiply by `gamma`.
            If False, `gamma` is not used.
            When the next layer is linear (also e.g. `nn.relu`),
            this can be disabled since the scaling
            will be done by the next layer.
        beta_initializer: Initializer for the beta weight.
        gamma_initializer: Initializer for the gamma weight.
        moving_mean_initializer: Initializer for the moving mean.
        moving_variance_initializer: Initializer for the moving variance.
        beta_regularizer: Optional regularizer for the beta weight.
        gamma_regularizer: Optional regularizer for the gamma weight.
        beta_constraint: Optional constraint for the beta weight.
        gamma_constraint: Optional constraint for the gamma weight.
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
    # Output shape
        Same shape as input.
    # References
        - [Batch Normalization: Accelerating Deep Network Training by
           Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167)
    """

    def __init__(self,
                 axis=-1,
                 momentum=0.99,
                 epsilon=1e-3,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 moving_mean_initializer='zeros',
                 moving_variance_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 **kwargs):
        self._trainable = True
        self._trainable_tensor = K.variable(1, dtype='float32', name='trainable')
        super(BatchNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.axis = axis
        self.momentum = momentum
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.moving_mean_initializer = initializers.get(moving_mean_initializer)
        self.moving_variance_initializer = (
            initializers.get(moving_variance_initializer))
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)

    @property
    def trainable(self):
        # Use cached value to avoid unnecessary get_value() calls
        return self._trainable

    @trainable.setter
    def trainable(self, trainable):
        trainable = bool(trainable)
        # Change when different to avoid unnecessary set_value() calls
        if self._trainable != trainable:
            self._trainable = trainable
            K.set_value(self._trainable_tensor, 1 if trainable else 0)

    def build(self, input_shape):
        dim = input_shape[self.axis]
        if dim is None:
            raise ValueError('Axis ' + str(self.axis) + ' of '
                                                        'input tensor should have a defined dimension '
                                                        'but the layer received an input with shape ' +
                             str(input_shape) + '.')
        self.input_spec = InputSpec(ndim=len(input_shape),
                                    axes={self.axis: dim})
        shape = (dim,)

        if self.scale:
            self.gamma = self.add_weight(shape=shape,
                                         name='gamma',
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint)
        else:
            self.gamma = None
        if self.center:
            self.beta = self.add_weight(shape=shape,
                                        name='beta',
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint)
        else:
            self.beta = None
        self.moving_mean = self.add_weight(
            shape=shape,
            name='moving_mean',
            initializer=self.moving_mean_initializer,
            trainable=False)
        self.moving_variance = self.add_weight(
            shape=shape,
            name='moving_variance',
            initializer=self.moving_variance_initializer,
            trainable=False)
        self.built = True

    def call(self, inputs, training=None):
        input_shape = K.int_shape(inputs)
        # Prepare broadcasting shape.
        ndim = len(input_shape)
        reduction_axes = list(range(len(input_shape)))
        del reduction_axes[self.axis]
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis]

        # Determines whether broadcasting is needed.
        needs_broadcasting = (sorted(reduction_axes) != list(range(ndim))[:-1])

        def normalize_inference():
            if needs_broadcasting:
                # In this case we must explicitly broadcast all parameters.
                broadcast_moving_mean = K.reshape(self.moving_mean,
                                                  broadcast_shape)
                broadcast_moving_variance = K.reshape(self.moving_variance,
                                                      broadcast_shape)
                if self.center:
                    broadcast_beta = K.reshape(self.beta, broadcast_shape)
                else:
                    broadcast_beta = None
                if self.scale:
                    broadcast_gamma = K.reshape(self.gamma,
                                                broadcast_shape)
                else:
                    broadcast_gamma = None
                return K.batch_normalization(
                    inputs,
                    broadcast_moving_mean,
                    broadcast_moving_variance,
                    broadcast_beta,
                    broadcast_gamma,
                    axis=self.axis,
                    epsilon=self.epsilon)
            else:
                return K.batch_normalization(
                    inputs,
                    self.moving_mean,
                    self.moving_variance,
                    self.beta,
                    self.gamma,
                    axis=self.axis,
                    epsilon=self.epsilon)

        if training in {0, False}:
            # If the learning phase is *static* and set to inference:
            return normalize_inference()
        elif training is None:
            # If it's undefined then if trainable tensor is on then
            # respect learning phase else set to false
            training = K.switch(self._trainable_tensor,
                                K.cast(K.learning_phase(), 'float32'),
                                K.constant(0, dtype='float32'))
            training._uses_learning_phase = True

        # If the learning is either dynamic, or set to training:
        normed_training, mean, variance = K.normalize_batch_in_training(
            inputs, self.gamma, self.beta, reduction_axes,
            epsilon=self.epsilon)

        if K.backend() != 'cntk':
            sample_size = K.prod([K.shape(inputs)[axis]
                                  for axis in reduction_axes])
            sample_size = K.cast(sample_size, dtype=K.dtype(inputs))

            # sample variance - unbiased estimator of population variance
            variance *= sample_size / (sample_size - (1.0 + self.epsilon))

        self.add_update([K.moving_average_update(self.moving_mean,
                                                 mean,
                                                 self.momentum),
                         K.moving_average_update(self.moving_variance,
                                                 variance,
                                                 self.momentum)],
                        inputs)

        # Pick the normalized form corresponding to the training phase.
        return K.in_train_phase(normed_training,
                                normalize_inference,
                                training=training)

    def get_config(self):
        config = {
            'axis': self.axis,
            'momentum': self.momentum,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'moving_mean_initializer':
                initializers.serialize(self.moving_mean_initializer),
            'moving_variance_initializer':
                initializers.serialize(self.moving_variance_initializer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
            'beta_constraint': constraints.serialize(self.beta_constraint),
            'gamma_constraint': constraints.serialize(self.gamma_constraint)
        }
        base_config = super(BatchNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape
'''

layers = tf.keras.layers


class BatchNormalization(layers.Layer):

    def __init__(self, momentum=0.999, epsilon=1e-3, **kwargs):
        super(BatchNormalization, self).__init__(**kwargs)
        self.momentum = momentum
        self.epsilon = epsilon

        self.scale = None
        self.beta = None
        self.pop_mean = None
        self.pop_var = None

    def build(self, inputs_shape):
        self.scale = self.add_weight(name='batch_norm_scale',
                                     shape=inputs_shape[1:],
                                     initializer=tf.keras.initializers.ones)
        self.beta = self.add_weight(name='batch_norm_beta',
                                    shape=inputs_shape[1:],
                                    initializer=tf.keras.initializers.zeros)
        self.pop_mean = self.add_weight(name='batch_norm_pop_mean',
                                        shape=inputs_shape[1:],
                                        initializer=tf.keras.initializers.zeros,
                                        trainable=False)
        self.pop_var = self.add_weight(name='batch_norm_pop_var',
                                       shape=inputs_shape[1:],
                                       initializer=tf.keras.initializers.ones,
                                       trainable=False)

        '''
        self.scale = tf.Variable(tf.ones([inputs_shape[-1]]))
        self.beta = tf.Variable(tf.zeros([inputs_shape[-1]]))
        self.pop_mean = tf.Variable(tf.zeros([inputs_shape[-1]]), trainable=False)
        self.pop_var = tf.Variable(tf.ones([inputs_shape[-1]]), trainable=False)
        '''

    def call(self, inputs, training=None):

        def batch_norm_train():
            batch_mean, batch_var = tf.nn.moments(inputs, [0])
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
