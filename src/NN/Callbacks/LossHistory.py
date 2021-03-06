from epicpath import EPath

import src.text.summary as summary
from .KerasCallback import KerasCallback


class LossHistory(KerasCallback):
    def __init__(self):
        super().__init__()
        self.logs = []  # logs = {'loss': 4.495124205389112, 'Output_0_loss' : 2.400269569744329,
        # 'Output_1_loss': 2.094854634212782, 'Output_0_acc_act': 0.9934636604502837,
        # 'Output_0_mae_dur': 0.2902308425676854, 'Output_1_acc_act': 0.9946330100062025,
        # 'Output_1_mae_dur': 0.25196381778232657}
        self.current_logs = None
        self.paths = []  # Where are stored the saved_models
        self.hparams = []  # the hyper parameters of the model

        self.best_index = None

        i = 0
        while EPath('tests_hp/Summary_test_{0}.txt'.format(i)).exists():
            i += 1
        self.path = EPath('tests_hp/Summary_test_{0}.txt'.format(i))
        EPath('tests_hp').mkdir(parents=True, exist_ok=True)
        with open(self.path.as_posix(), 'w') as f:
            f.write('\n')

    def on_train_begin(self, logs={}):
        self.current_logs = None

    def on_epoch_end(self, epoch, logs={}):
        self.current_logs = logs

    def on_train_end(self, logs=None):
        self.logs.append(self.current_logs)
        # Update the best index
        if len(self.logs) == 1:
            self.best_index = 0
        elif LossHistory.better_than(self.logs[-1], self.logs[self.best_index]):
            self.best_index = len(self.logs) - 1

    # ------ Personal methods ------
    def find_best_index(self):
        best_index = 0
        for i in range(1, len(self.logs)):
            if LossHistory.better_than(self.logs[best_index], self.logs[i]):
                best_index = i
        self.best_index = best_index
        return self.best_index

    @staticmethod
    def better_than(d1, d2):
        # Global loss
        if d1['loss'] != d2['loss']:
            return d1['loss'] < d2['loss']

        # Square of single loss
        l1, l2 = 0, 0
        i = 0
        key = 'Output_{0}_loss'.format(i)
        while key in d1 and key in d2:
            l1 += d1[key] ** 2
            l2 += d2[key] ** 2
            i += 1
            key = 'Output_{0}_loss'.format(i)
        if l1 != l2:
            return l1 < l2

        # Accuracy
        a1, a2 = 0, 0
        i = 0
        key = 'Output_{0}_acc_act'.format(i)
        while key in d1 and key in d2:
            a1 += d1[key]
            a2 += d2[key]
            i += 1
            key = 'Output_{0}_acc_act'.format(i)
        return a1 >= a2

    def update_summary(self, i=None):
        summary.update_summary_loss_history(self.path.as_posix(), self.logs[-1], self.paths[-1], self.hparams[-1], i)

    def update_best_summary(self):
        summary.update_best_summary_loss_history(self.path.as_posix(), self.logs[self.best_index],
                                                 self.paths[self.best_index], self.hparams[self.best_index],
                                                 self.best_index)
