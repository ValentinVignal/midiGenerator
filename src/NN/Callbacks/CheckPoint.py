import numpy as np
import tensorflow as tf
from pathlib import Path

from .KerasCallback import KerasCallback


class CheckPoint(KerasCallback):
    """

    """
    def __init__(self, filepath):
        """

        :param filepath:
        """
        self.filepath = Path(filepath)
        super(CheckPoint, self).__init__()

        self.best_acc = -np.Inf
        self.best_loss = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        """

        :param epoch:
        :param logs:
        :return:
        """
        logs = logs or {}
        if self.greater_result(logs):
            self.best_acc = self.get_val_acc_mean(logs)
            self.best_loss = self.get_val_loss(logs)
            # self.model.save_weights(self.filepath, overwrite=True)
            if self.filepath.exists():
                self.filepath.unlink()
            tf.keras.experimental.export_saved_model(self.model, self.filepath.as_posix())

    def greater_result(self, logs):
        """

        :param logs:
        :return:
        """
        acc = self.get_val_acc_mean(logs)
        return np.greater(
            acc, self.best_acc
        ) or (
                np.equal(acc, self.best_acc)
                and
                np.less(self.get_val_loss(logs), self.best_loss)
        )

    @staticmethod
    def get_val_acc_mean(logs):
        """

        :param logs:
        :return:
        """
        i = 0
        res = 0
        while f'val_Output_{i}_acc' in logs:
            res += logs.get(f'val_Output_{i}_acc')
            i += 1
        return res / i

    @staticmethod
    def get_val_loss(logs):
        """

        :param logs:
        :return:
        """
        return logs.get('loss_val')



