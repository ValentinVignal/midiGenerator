import numpy as np

from .KerasCallback import KerasCallback


class AddAcc(KerasCallback):
    """
    So there is always _acc metric in logs
    """
    def __init__(self, mono=False):
        """

        :param mono:
        """
        self.mono = mono
        super(AddAcc, self).__init__()

    def on_epoch_end(self, epochs, logs={}):
        """

        :param epochs:
        :param logs:
        :return:
        """
        if 'val_Output_0_acc' not in logs:
            # So we have to construct it
            if self.mono:
                i = 0
                while f'val_Output_{i}_loss' in logs:
                    acc = np.mean([
                        logs[f'val_Output_{i}_acc_cat'],
                        logs[f'val_Output_{i}_acc_bin']
                    ])
                    logs[f'val_Output_{i}_acc'] = acc
                    i += 1
        if 'Output_0_acc' not in logs:
            # So we have to construct it
            if self.mono:
                i = 0
                while f'Output_{i}_loss' in logs:
                    acc = np.mean([
                        logs[f'Output_{i}_acc_cat'],
                        logs[f'Output_{i}_acc_bin']
                    ])
                    logs[f'Output_{i}_acc'] = acc
                    i += 1
