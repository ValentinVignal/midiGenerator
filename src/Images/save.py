from PIL import Image
import numpy as np
from matplotlib import pyplot as plt


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


