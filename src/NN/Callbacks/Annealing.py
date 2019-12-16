import tensorflow as tf

from .KerasCallback import KerasCallback

K = tf.keras.backend


class Annealing(KerasCallback):
    def __init__(self, weight, start_value=0, final_value=1, epoch_start=0.4, epoch_stop=0.8):
        """

        :param weight: the K.variable
        :param start_value: the value to start
        :param final_value: the value to have at the end of the training
        :param epoch_start: the number of epoch to wait
            - before : weigth = 0
            - the type:
                - int: the absolute number
                - float: the portion
                ex:
                1)
                nb_epochs = 100
                epoch_start = 20
                    => before epoch 20: weight = 0
                ----------
                2)
                nb_epochs = 100
                epoch_start = 0.25
                    => before epoch 25: weight = 0
        :param epoch_stop: same as epoch_start except that if epoch > epoch_stop:
            => weight = final_value

        """
        self.weigth = weight
        self.epochs = None
        self.start_value = start_value
        self.final_value = final_value
        self.epoch_start = epoch_start
        self.epoch_stop = epoch_stop

        self.delta = None

    def on_epoch_begin(self, epoch, logs={}):
        """

        :param epoch:
        :param logs:
        :return:
        """
        if epoch < self.epoch_start:
            K.set_value(self.weigth, self.start_value)
        elif epoch >= self.epoch_start:
            K.set_value(self.weigth, self.start_value + (epoch - self.epoch_start) * self.delta)
        elif epoch > self.epoch_stop:
            K.set_value(self.weigth, self.final_value)

    def compute_epoch_start(self):
        if isinstance(self.epoch_start, float):
            self.epoch_start = int(self.epoch_start * self.epochs)

    def compute_epoch_stop(self):
        if isinstance(self.epoch_stop, float):
            self.epoch_stop = int(self.epoch_stop * self.epochs)

    def update_with_fit_args(self, **kwargs):
        self.epochs = kwargs['epochs']
        self.compute_epoch_start()
        self.compute_epoch_stop()
        nb_active_epochs = self.epoch_stop - self.epoch_start
        if nb_active_epochs > 1:
            self.delta = (self.final_value - self.start_value) / (nb_active_epochs - 1)
        else:
            self.delta = 0

