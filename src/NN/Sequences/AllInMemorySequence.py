"""
A sequence with all the data in memory
"""
import numpy as np
import tensorflow as tf


class AllInMemorySequence(tf.keras.utils.Sequence):
    """

    """

    def __init__(self, sequence, batch_size=None):
        """

        :param sequence: KerasSequence
        :param batch_size:
        """
        self.batch_size = sequence.batch_size if batch_size is None else batch_size
        self.x = None
        self.y = None
        self.nb_steps = None

        self.load(sequence)

    def load(self, sequence):
        print('Loading all the data in an AllInMemorySequence instance')
        old_batch_size = sequence.batch_size
        sequence.change_batch_size(1)
        self.nb_steps = len(sequence)

        first_x, first_y = sequence[0]
        nb_inputs = len(first_x)
        nb_outputs = len(first_y)
        all_x = [[] for _ in range(nb_inputs)]
        all_y = [[] for _ in range(nb_outputs)]
        for i in range(len(sequence)):
            x, y = sequence[i]
            for j in range(nb_inputs):
                all_x[j].append(x[j])
            for j in range(nb_outputs):
                all_y[j].append(y[j])
        self.x = [np.concatenate(l, axis=0) for l in all_x]
        self.y = [np.concatenate(l, axis=0) for l in all_y]
        sequence.change_batch_size(old_batch_size)

    def __len__(self):
        return self.nb_steps // self.batch_size

    def __getitem__(self, item):
        return [x[item:item + self.batch_size] for x in self.x], [y[item:item + self.batch_size] for y in self.y]
