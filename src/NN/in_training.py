import tensorflow as tf
from src.NN.data_generator import MySequence
import numpy as np


class WrappedInTrainingModel(tf.keras.Model):
    def __init__(self, **kwargs):
        self.in_training = None
        super(WrappedInTrainingModel, self).__init__(**kwargs)

    def update_in_training(self, in_training, default_in_training=1):
        if in_training is not None:
            self.in_training = in_training
        if self.in_training is None:
            self.in_training = default_in_training

    def new_inputs(self, inputs):
        if type(inputs) is list:
            batch = len(inputs[0])
            input_learning = np.ones(batch) * self.in_training
            inputs.append(input_learning)
        else:
            batch = len(inputs)
            input_learning = np.ones(batch) * self.in_training
            inputs = [inputs, input_learning]
        return (inputs)

    def fit(self, x=None, y=None, in_training=1, **kwargs):
        self.update_in_training(in_training, 1)  # in_training = None to keep previous state
        x = self.new_inputs(x)
        return super(WrappedInTrainingModel, self).fit(x=x, y=y, **kwargs)

    def evaluate(self, x=None, y=None, in_training=0, **kwargs):
        self.update_in_training(in_training, default_in_training=0)
        x = self.new_inputs(x)
        return super(WrappedInTrainingModel, self).evaluate(x=x, y=y, **kwargs)

    def predict(self, inputs, in_training=0, **kwargs):
        self.update_in_training(in_training, default_in_training=0)
        inputs = self.new_inputs(inputs)
        return super(WrappedInTrainingModel, self).predict(inputs, **kwargs)

    def fit_generator(self, generator, in_training=1, **kwargs):
        self.update_in_training(in_training, default_in_training=1)
        generator.set_in_training(self.in_training)
        return super(WrappedInTrainingModel, self).fit_generator(generator=generator, **kwargs)

    def evaluate_generator(self, generator, in_training=0, **kwargs):
        self.update_in_training(in_training, default_in_training=0)
        generator.set_in_training(self.in_training)
        return super(WrappedInTrainingModel, self).evaluate_generator(generator=generator, **kwargs)


class WrapInTrainingMySequence(MySequence):
    def __init__(self, **kwargs):
        self.in_training = None
        super(WrapInTrainingMySequence, self).__init__(**kwargs)

    def set_in_training(self, in_training):
        self.in_training = in_training

    def new_inputs(self, inputs):
        if type(inputs) is list:
            batch = len(inputs[0])
            input_learning = np.ones(batch) * self.in_training
            inputs.append(input_learning)
        else:
            batch = len(inputs)
            input_learning = np.ones(batch) * self.in_training
            inputs = [inputs, input_learning]
        return (inputs)

    def __getitem__(self, index):
        x, y = super(WrapInTrainingMySequence, self).__getitem__(index)
        x = self.new_inputs(x)
        return x, y

    def get_inputs(self, index):
        return super(WrapInTrainingMySequence, self).__getitem__(index)