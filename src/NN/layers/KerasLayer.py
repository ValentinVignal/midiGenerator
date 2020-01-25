import tensorflow as tf

layers = tf.keras.layers


class KerasLayer(layers.Layer):
    """
    Wrapper class Layer of Keras
    """
    def __init__(self, *args, **kwargs):
        super(KerasLayer, self).__init__(*args, **kwargs)

    @classmethod
    def from_config(cls, *args, **kwargs):
        print('args', args)
        print('kwargs', kwargs)
        return super(KerasLayer, cls).from_config(*args, **kwargs)

    def reset_weights_variables(self):
        """
        Used to reset the weights of a sublayers in a layers

        Since Tensorflow 2.0 it works without this function

        :return:
        """
        # self._trainable_weights = []
        # self._non_trainable_weights = []
        # self._trainable_variables = []
        # self._non_trainable_variables = []
        pass

    def add_weights_variables(self, *args):
        """
        Used to add the weights of a sublayers in a layers

        Since Tensorflow 2.0 it works without this function

        :param args:
        :return:
        """
        # for l in args:
        #     self._trainable_weights += l.trainable_weights
        #     self._non_trainable_weights += l.non_trainable_weights
        #     self._trainable_variables += l.trainable_variables
        #     self._non_trainable_variables += l.non_trainable_variables
        pass

    def set_weights_variables(self, *args):
        """
        Used to set the weights of a sublayers in a layers

        Since Tensorflow 2.0 it works without this function

        :param args:
        :return:
        """
        # self.reset_weights_variables()
        # self.add_weights_variables(*args)
        pass

    def __call__(self, *args, **kwargs):
        return super(KerasLayer, self).__call__(*args, **kwargs)

    def build(self, inputs_shape):
        super(KerasLayer, self).build(inputs_shape)
        # print(self.__class__.__name__, 'is built with shape', inputs_shape)

    def update_with_fit_args(self, **kwargs):
        """

        :param kwargs:
        :return:
        """
        pass

    def on_epoch_begin(self, *ags, **kwags):
        """

        :param ags:
        :param kwags:
        :return:
        """
        pass


