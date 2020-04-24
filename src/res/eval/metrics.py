import numpy as np
import math

from src import GlobalVariables as g


def harmony(array, l1=g.loss.l_semitone, l2=g.loss.l_tone, l6=g.loss.l_tritone):
    """

    :param array: (instruments, size, length, channels)
    :param l1:
    :param l2:
    :param l6:
    :return: (length,)
    """
    arr1 = harmony_n(array, 1)
    arr2 = harmony_n(array, 2)
    arr6 = harmony_n(array, 6)
    value = l1 * arr1 + l2 * arr2 + l6 * arr6       # (length,)
    return value


def harmony_n(array, n):
    """

    :param array: (instruments, size, length, channels)
    :param n:
    :return: (length,)
    """
    activations = np.take(array, axis=-1, indices=0)       # (instruments, size, length)
    arr_sum = np.sum(
        a=activations,
        axis=0
    )       # (size, length)
    nb_notes, length = arr_sum.shape
    nb_notes_rounded = int(math.ceil(nb_notes / 12)) * 12
    activations = np.zeros((nb_notes_rounded, length))
    activations[-nb_notes:] = arr_sum
    per_notes = np.reshape(activations, newshape=(12, -1, length))      # (12, nb_octaves, length)
    per_notes = np.sum(per_notes, axis=0)       # (12, length)
    rolled = np.roll(
        a=per_notes,
        shift=n,
        axis=0
    )
    harmony_n_value = np.sum(
        a=per_notes * rolled,
        axis=0
    )       # (length,)
    length_rounded = int(math.ceil(len(harmony_n_value) / 16)) * 16
    harmony_arr_rounded = np.zeros(length_rounded)
    harmony_arr_rounded[:len(harmony_n_value)] = harmony_n_value
    harmony_measure = np.reshape(harmony_arr_rounded, newshape=(-1, 16))        # (nb_measures, 16)
    return np.sum(harmony_measure, axis=1)








