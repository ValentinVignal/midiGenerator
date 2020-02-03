import numpy as np

from .KerasCallback import KerasCallback


class BestAccuracy(KerasCallback):
    """

    """

    def __init__(self):
        """

        """
        super(BestAccuracy, self).__init__()

        self.best_acc = -np.Inf
        self.best_loss = np.Inf
        self.best_epoch = None
        self.best_logs = None

    def on_epoch_end(self, epoch, logs=None):
        """

        :param epoch:
        :param logs:
        :return:
        """
        # print('epoch', epoch)
        # print('self.best_acc', self.best_acc, 'self.best_loss', self.best_loss)
        # print('acc epoch', self.get_val_acc_mean(logs), 'loss epoch', self.get_val_loss(logs))
        logs = logs or {}
        if self.greater_result(logs):
            self.best_acc = self.get_val_acc_mean(logs)
            self.best_loss = self.get_val_loss(logs)
            self.best_epoch = epoch
            self.best_logs = logs

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
        return logs.get('val_loss')
