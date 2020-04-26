import numpy as np
from epicpath import EPath
import os
import sys
import pickle


if __name__ == '__main__':
    sys.path.append(os.path.abspath('.'))

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
        h_s = res['harmony_seed']
        h_c = res['harmony_created']
        if h_s is not None and not np.isnan(h_s):
            res_seed += res['harmony_seed']
        if h_c is not None and not np.isnan(h_c):
            res_created += h_c
        text += file.rstem + '\n'
        for k in res:
            text += f'\t{k}: {res[k]}\n'
        plots.plot_res((file.parent / file.rstem + '_harmony_measure').with_suffix('.jpg'), res)
    res_seed /= len(files)
    res_created /= len(files)
    text = f'\t\tRes for generation\n\nMean:\n\tharmony_seed: {res_seed}\n\tharmony_created: {res_created}\n' + text
    with open(path / 'res.txt', 'w') as f:
        f.write(text)
    with open(path / 'res.p', 'wb') as dump_file:
        pickle.dump(all_res, dump_file)


if __name__ == '__main__':
    path = EPath('..', '..', '..', 'big_runs', 'lstm')
    for folder in path.listdir(concat=True):
        if not folder.is_dir():
            continue
        midi_folder = folder / 'generated_midis'
        midi_folder /= midi_folder.listdir(concat=False)[0]
        eval_songs_folder(midi_folder)














