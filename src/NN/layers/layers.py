import tensorflow as tf

layers = tf.keras.layers


class KerasLayer(layers.Layer):
    """
    Wrapper class Layer of Keras
    """
    def __init__(self, *args, **kwargs):
        super(KerasLayer, self).__init__(*args, **kwargs)

    def reset_weights_variables(self):
        self._trainable_weights = []
        self._non_trainable_weights = []
        self._trainable_variables = []
        self._non_trainable_variables = []

    def add_weights_variables(self, *args):
        for l in args:
            self._trainable_weights += l.trainable_weights
            self._non_trainable_weights += l.non_trainable_weights
            self._trainable_variables += l.trainable_variables
            self._non_trainable_variables += l.non_trainable_variables

    def set_weights_variables(self, *args):
        self.reset_weights_variables()
        self.add_weights_variables(*args)
