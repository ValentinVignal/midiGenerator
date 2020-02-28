import numpy as np

from . import colors as mcolors
from . import save as msave

from src import Midi


def array_to_pianoroll(array, seed_length=0, mono=False, replicate=False, colors=None, notes_range=None,
                       step_length=None):
    """

    :param step_length:
    :param notes_range: To add the piano on the left of the image
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
            all[:, i::2 * seed_length] = 20  # So steps are visible (grey)
    else:
        all[:, :seed_length] = 30  # So seed is visible (grey)
    if step_length:
        # Show the steps:
        all[:, ::step_length] += 10
        all[:, ::step_length, 2] += 10
    if notes_range is not None:
        # Put the white keys whiter
        for note in range(*notes_range):
            if note % 12 not in [1, 4, 6, 9, 11]:  # (A#, C#, D#, F#, G#)
                all[note - notes_range[0]] += 20
    for inst in range(nb_instruments):
        for i in range(input_size):
            for j in range(nb_steps):
                if activations[inst, i, j] == 1:
                    all[i, j] = colors[inst]
    all = np.flip(all, axis=0)      # (input_size, length, 3)

    if notes_range is not None:
        # Add the piano on the left of the image
        all = np.concatenate([piano(notes_range), all], axis=1)
    return all


def save_array_as_pianoroll(array, folder_path, name, seed_length=0, mono=False, replicate=False, save_pil=True,
                            save_plt=True, notes_range=None, step_length=None):
    """

    :param notes_range: To add the piano on the left of the image
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
        replicate=replicate,
        notes_range=notes_range,
        step_length=step_length
    )
    msave.save_array(
        array=pianoroll,
        folder_path=folder_path,
        name=name,
        save_pil=save_pil,
        save_plt=save_plt
    )


def save_arrays_as_pianoroll_subplot(arrays, file_name, titles=None, subtitles=None, seed_length=0, mono=False,
                                     replicate=False, notes_range=None, step_length=None):
    """

    :param notes_range: To add the image of the piano on the left
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
        colors=colors,
        notes_range=notes_range,
        step_length=step_length
    ) for array in arrays]
    msave.save_arrays(
        arrays=pianorolls,
        file_name=file_name,
        titles=titles,
        subtitles=subtitles,
    )


def piano(notes_range, notes_range_type='model'):
    """

    :param notes_range: note range of the model
    :param notes_range_type: 'model' (default) / 'midi'
    :return: The image RGB of the piano
    """
    if notes_range_type == 'midi':
        notes_range = Midi.create.midinote_to_note(notes_range[0]), Midi.create.midinote_to_note(notes_range[1])
    piano = 255 * np.ones((notes_range[1] - notes_range[0], 4, 3), dtype=np.int)       # (input_size, l, 3)
    for note in range(*notes_range):
        if note % 12 in [1, 4, 6, 9, 11]:   # (A#, C#, D#, F#, G#)
            piano[note - notes_range[0]] = 0
    return np.flip(piano, axis=0)




