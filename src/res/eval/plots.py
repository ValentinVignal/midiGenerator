import matplotlib.pyplot as plt
import numpy as np


def plot_res(filename, res):
    """

    :param filename:
    :param res:
    :return:
    """
    harmony_measure = res['harmony_measure']
    plt.figure()
    plt.title(filename.stem + ' harmony_measure')
    plt.xlabel('measure')
    plt.ylabel('harmony value')
    plt.plot(
        np.arange(1, len(harmony_measure) + 1),
        harmony_measure
    )
    plt.grid()
    plt.savefig(filename)
    plt.close()
