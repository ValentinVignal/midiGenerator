import tensorflow as tf
import warnings
import numpy as np


class KerasModel(tf.keras.Model):
    """
    Wrapper of the class tf.keras.Model
    """
    def __init__(self, *args, **kwargs):
        super(KerasModel, self).__init__(*args, **kwargs)
        self.already_fit = False

    def build(self, input_shape, *args, **kwargs):
        super(KerasModel, self).build(input_shape, *args, **kwargs)

    def reset_weights_variables(self):
        """
        Used to reset the weights of a sublayers in a model

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
        Used to add the weights of a sublayers in a model

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
        Used to set the weights of a sublayers in a model

        Since Tensorflow 2.0 it works without this function

        :param args:
        :return:
        """
        # self.reset_weights_variables()
        # self.add_weights_variables(*args)
        pass

    def fit(self, *args, **kwargs):
        self.already_fit = True
        return super(KerasModel, self).fit(*args, **kwargs)

    def fit_generator(self, generator, *args, **kwargs):
        """
        Before TF2, it was necessary to fit the model before fit_generator
        But now it works

        :param generator:
        :param args:
        :param kwargs:
        :return:
        """
        # if not self.already_fit:
        #     x, y = generator[0]
        #     self.fit(x=x, y=y, verbose=0)

        return super(KerasModel, self).fit_generator(generator=generator, *args, **kwargs)

    def generate(self, x, *args, **kwargs):
        """
        to filter messy output if there are some

        :param x:
        :param args:
        :param kwargs:
        :return:
        """
        return self.predict(x=x, *args, **kwargs)

    def print_summary(self):
        """
        Print a "working" summary
        :return: nothing
        """
        try:
            print(self.summary())
        except ValueError:
            warnings.warn(f'The model {self.name} has not yet been built, impossible to print the summary')

    def summary(self, *args, **kwargs):
        try:
            return super(KerasModel, self).summary(*args, **kwargs)
        except ValueError:
            warnings.warn(f'The model {self.name} has not yet been built, impossible to print the summary')
            return None


def fake_input(input_shape):
    if isinstance(input_shape, list):
        inputs = []
        for _input_shape in input_shape:
            input_shape_ = []
            for i_s in _input_shape:
                i_s_ = 1 if i_s is None else i_s
                input_shape_.append(i_s_)
            inputs.append(np.zeros(tuple(input_shape_), dtype=np.float32))
        return inputs

    else:
        input_shape_ = []
        for i_s in input_shape:
            i_s_ = 1 if i_s is None else i_s
            input_shape_.append(i_s_)
        input_shape_ = tuple(input_shape_)
        return np.zeros(input_shape_, dtype=np.float32)


