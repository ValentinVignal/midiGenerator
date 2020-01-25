import tensorflow as tf

from .KerasCallback import KerasCallback

K = tf.keras.backend


class UpdateLayers(KerasCallback):
    def __init__(self, *args, **kwargs):
        self.fit_args = []
        self.fit_kwargs = {}
        super(UpdateLayers, self).__init__(*args, **kwargs)

    def on_epoch_begin(self, epoch, logs={}):
        """

        :param epoch:
        :param logs:
        :return:
        """
        for layer in self.model:
            layer.on_epoch_begin(epoch, logs)

    def update_with_fit_args(self, *args, **kwargs):
        self.fit_args = args
        self.fit_kwargs = kwargs

    def on_train_begin(self, logs):
        print(dir(self.model))
        for layer in self.model.layers:
            layer.update_with_fit_args(*self.args, **self.kwargs)



