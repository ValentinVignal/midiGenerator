import tensorflow as tf
import math
import numpy as np


class MySequenceReduced(tf.keras.utils.Sequence):
    """
    This class is use to train with a validation split
    """
    def __init__(self, my_sequence, indexes):
        self.my_sequence = my_sequence
        self.indexes = indexes

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, index):
        return self.my_sequence[self.indexes[index]]


def get_train_valid_mysequence(my_sequence, validation_split=0.0):
    """

    :param my_sequence:
    :param validation_split:
    :return:
    """
    all_len = len(my_sequence)
    len_valid = math.ceil(all_len * validation_split)
    permutation = np.random.permutation(np.arange(all_len))
    indexes_valid = permutation[:len_valid]
    indexes_train = permutation[len_valid:]
    my_sequence_valid = MySequenceReduced(my_sequence, indexes_valid)
    my_sequence_train = MySequenceReduced(my_sequence, indexes_train)
    return my_sequence_train, my_sequence_valid
