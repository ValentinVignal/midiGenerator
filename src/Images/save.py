from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from termcolor import cprint


def save_array(array, folder_path, name, save_pil=True, save_plt=True):
    """

    :param save_pil:
    :param save_plt:
    :param array: (a, b, 3)         # RGB
    :param folder_path:
    :param name:
    :return:
    """
    if save_pil:
        file_name = folder_path / (name + '_(PIL).jpg')
        save_array_pil(file_name, array)
    if save_plt:
        file_name = folder_path / (name + '_(PLT).jpg')
        save_array_plt(file_name, array)


def save_array_pil(file_name, array):
    """

    :param file_name:
    :param array: (a, b, 3)         # RGB
    :return:
    """
    Image.fromarray(array.astype(np.uint8), mode='RGB').save(file_name)


def save_array_plt(file_name, array):
    """

    :param file_name:
    :param array: (a, b, 3)         # RGB
    :return:
    """
    plt.imshow(array.astype(np.int))
    plt.title(file_name.stem)
    plt.savefig(file_name)


def save_arrays(arrays, file_name, titles=None, subtitles=None):
    """

    :param arrays: List[(a, b, 3)]
    :param file_name:
    :param titles:
    :param subtitles:
    :return:
    """
    nb_arrays = len(arrays)
    fig, axs = plt.subplots(nb_arrays, 1)
    for i in range(nb_arrays):
        axs[i].imshow(arrays[i].astype(np.int))
        if titles is not None:
            title = titles[i]
            if subtitles is not None:
                title += f'\n{subtitles[i]}'
            axs[i].set_title(title)
    plt.savefig(file_name)



