import math
import numpy as np

from src import GlobalVariables as g


def range_notes_in_matrix(matrix, mono=False):
    """

    :param mono:
    :param matrix: (nb_instruments, input_size, max_length, 2)
    :return: max and min note
    """
    matrix = matrix[:, :-1] if mono else matrix
    m, M = 0, matrix.shape[1]
    while m < M and np.all(matrix[:, m] == 0):
        m += 1
    while m < M and np.all(matrix[:, M - 1] == 0):
        M -= 1
    return m, M


def nb_measures(matrix):
    """

    :param matrix: (nb_instruments, input_size, max_lenght, 2)
    :return:
    """
    nb_steps = matrix.shape[2]
    nb_m = math.floor(nb_steps / (4 * g.midi.step_per_beat))
    return nb_m

