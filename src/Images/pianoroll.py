import numpy as np

from . import colors as mcolors
from . import save as msave


def array_to_pianoroll(array, seed_length=0, mono=False, replicate=False, colors=None):
    """

    :param colors:
    :param array: (nb_instruments, input_size, nb_steps, channels)
    :param seed_length:
    :param mono:
    :param replicate:
    :return: The numpy array corresponding ton the image of the pianoroll
    """
    nb_instruments = array.shape[0]
    # Get the colors for the instruments
    colors = mcolors.get_colors(nb_instruments) if colors is None else colors
    activations = np.take(array, axis=-1, indices=0)  # (nb_instruments, size, nb_steps)
    activations = activations[:, :-1] if mono else activations
    nb_instruments, input_size, nb_steps = activations.shape
    np.place(activations, 0.5 <= activations, 1)
    np.place(activations, activations < 0.5, 0)
    all = np.zeros((input_size, nb_steps, 3))  # RGB        (input_size, length, 3)
    if replicate:
        for i in range(seed_length):
            all[:, i::2 * seed_length] = 25  # So steps are visible (grey)
    else:
        all[:, :seed_length] = 25  # So seed is visible (grey)
    for inst in range(nb_instruments):
        for i in range(input_size):
            for j in range(nb_steps):
                if activations[inst, i, j] == 1:
                    all[i, j] = colors[inst]
    all = np.flip(all, axis=0)      # (input_size, length, 3)
    return all


def save_array_as_pianoroll(array, folder_path, name, seed_length=0, mono=False, replicate=False, save_pil=True,
                            save_plt=True):
    """

    :param array: (nb_instruments, size, length, channels)
    :param folder_path:
    :param name:
    :param seed_length:
    :param mono:
    :param replicate:
    :param save_pil:
    :param save_plt:
    :return:
    """

    pianoroll = array_to_pianoroll(
        array=array,
        seed_length=seed_length,
        mono=mono,
        replicate=replicate
    )
    msave.save_array(
        array=pianoroll,
        folder_path=folder_path,
        name=name,
        save_pil=save_pil,
        save_plt=save_plt
    )


def save_arrays_as_pianoroll_subplot(arrays, file_name, titles=None, subtitles=None, seed_length=0, mono=False,
                                     replicate=False):
    """

    :param subtitles:
    :param titles:
    :param arrays: List[(nb_instruments, size, length, channels)]
    :param file_name:
    :param seed_length:
    :param mono:
    :param replicate:
    :return:
    """
    nb_instruments = arrays[0].shape[0]
    colors = mcolors.get_colors(nb_instruments)
    pianorolls = [array_to_pianoroll(
        array=array,
        seed_length=seed_length,
        mono=mono,
        replicate=replicate,
        colors=colors
    ) for array in arrays]
    msave.save_arrays(
        arrays=pianorolls,
        file_name=file_name,
        titles=titles,
        subtitles=subtitles
    )
