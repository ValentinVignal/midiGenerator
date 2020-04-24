import numpy as np
from epicpath import EPath
import math

from src import Midi

from . import metrics
from . import plots


def eval_song_array(array, seed_length=None):
    """

    :param seed_length:
    :param array: (instruments, size, length, channels)
    :return:
    """
    harmony_arr = metrics.harmony(array)  # (nb_measures,)
    if seed_length is None:
        # There is no seed
        harmony_seed = None
        harmony_created = np.mean(harmony_arr)
    else:
        # There is a seed
        harmony_seed = np.mean(harmony_arr[:seed_length])
        harmony_created = np.mean(harmony_arr[seed_length:])

    res = dict(
        harmony_seed=harmony_seed,
        harmony_created=harmony_created,
        harmony_measure=harmony_arr
    )
    return res


def eval_song_midi(path, seed_length=None):
    """

    :param path:
    :return:
    """
    array = Midi.open.midi_to_matrix_bach(
        filename=path,
        length=None,
        notes_range=None,
        transpose=False
    )   # (nb_instruments, input_size, nb_steps, channels)
    # array_mono = Midi.open.to_mono_matrix(array)        # (nb_instruments, input_size, nb_steps, channels)
    res = eval_song_array(array[:, :, :, [0]], seed_length=seed_length)
    return res


def eval_songs_folder(path):
    """

    :param path:
    :return:
    """
    path = EPath(path)
    files = list(filter(
        lambda x: x.suffix == '.mid',
        path.listdir(concat=True)
    ))
    all_res = {}
    text = ""
    res_seed = 0
    res_created = 0
    for file in files:
        seed_length = 8 if file.rstem[20:] != "redo_song_generate_3" else None
        res = eval_song_midi(file, seed_length)
        all_res[file.rstem] = res
        res_seed += res['harmony_seed']
        res_created += res['harmony_created']
        text += file.rstem + '\n'
        for k in res:
            text += f'\t{k}: {res[k]}\n'
        plots.plot_res((file.parent / file.rstem + '_harmony_measure').with_suffix('.jpg'), res)
    res_seed /= len(files)
    res_created /= len(files)
    text = f'\t\tRes for generation\n\nMean:\n\tharmony_seed: {res_seed}\n\tharmony_created: {res_created}\n' + text
    with open(path / 'res.txt', 'w') as f:
        f.write(text)








