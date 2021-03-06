import numpy as np
import music21
from termcolor import colored, cprint

from . import instruments as midi_inst
from src import GlobalVariables as g


def normalize_activation(arr, threshold=0.5, mono=False, use_binary=g.loss.use_binary):
    """

    :param use_binary:
    :param mono:
    :param arr: (nb_instruments, (batch=1), nb_steps=1, length, 88, 2)
    :param threshold:
    :return: the same array but only with one and zeros for the activation part ([:, :, :, 0])
    """
    if mono:
        if use_binary:
            if len(arr.shape) == 5:
                np.expand_dims(arr, axis=1)  # (nb_instruments, batch, nb_steps, length, size, channels)
            activation_binary = arr[:, :, :, :, -1:]
            np.place(activation_binary, threshold <= activation_binary, 1)
            np.place(activation_binary, activation_binary < threshold, 0)

            cat = arr[:, :, :, :, :-1]
            axis = len(cat.shape) - 2
            argmax = np.argmax(cat, axis=axis)
            new_cat = np.zeros_like(cat)
            idx = list(np.ogrid[[slice(cat.shape[ax]) for ax in range(cat.ndim) if ax != axis]])
            idx.insert(axis, argmax)
            new_cat[tuple(idx)] = 1

            new_arr = np.zeros_like(arr)
            new_arr[:, :, :, :, :-1] = new_cat * (1 - activation_binary)
            new_arr[:, :, :, :, -1:] = activation_binary
        else:
            axis = len(arr.shape) - 2
            argmax = np.argmax(arr, axis=axis)
            new_arr = np.zeros(arr.shape)
            idx = list(np.ogrid[[slice(arr.shape[ax]) for ax in range(arr.ndim) if ax != axis]])
            idx.insert(axis, argmax)
            new_arr[tuple(idx)] = 1
    else:
        activation_binary = np.take(arr, axis=-1, indices=0)
        np.place(activation_binary, threshold <= activation_binary, 1)
        np.place(activation_binary, activation_binary < threshold, 0)
        np.put_along_axis(
            arr=arr,
            indices=np.zeros(tuple(1 for i in arr.shape), dtype=int),
            values=np.expand_dims(activation_binary, axis=-1),
            axis=-1
        )
        new_arr = arr
    return new_arr


def converter_func_poly(arr, no_duration=False):
    """

    :param arr: (nb_instruments, 88, nb_steps, 2)
    :param no_duration: if True : all notes will be the shortest length possible
    :return:
    """
    activations = arr[:, :, :, 0]
    durations = arr[:, :, :, 1]

    np.place(activations, 0.5 <= activations, 1)
    np.place(activations, activations < 0.5, 0)

    durations = np.ceil(durations * g.midi.max_length_note_array)
    durations = np.maximum(durations, 1)

    # If no duration then no nee to compute duration and return activation (all durations = 1)
    if no_duration:
        return activations

    # Else we have to compute the durations
    matrix_norm = np.multiply(activations, durations)  # (nb_instruments, 128, nb_steps)

    # ----- ----- ----- ----- ----- ----- ----- ----- ----- -----

    nb_instruments, nb_notes, nb_steps = matrix_norm.shape
    for instrument in range(nb_instruments - 1, -1, -1):
        for note in range(nb_notes - 1, -1, -1):
            duration = 1
            for step in range(nb_steps - 1, -1, -1):
                if matrix_norm[instrument, note, step] == 0:
                    duration += 1
                else:
                    matrix_norm[instrument, note, step] = min(matrix_norm[instrument, note, step], duration)
                    duration = 1

    return matrix_norm


def converter_func_mono(arr, use_binary=g.loss.use_binary):
    """

    :param use_binary:
    :param arr: (nb_instruments, 88, nb_steps, 1)
    :return:
    """
    arr = arr[:, :, :, 0]
    if use_binary:
        activations = arr[:, -1:]
        np.place(activations, 0.5 <= activations, 1)
        np.place(activations, activations < 0.5, 0)
        cat = arr[:, :-1]
    else:
        cat = arr

    argmax = np.argmax(cat, axis=1)
    cat_norm = np.zeros_like(cat)  # (nb_instruments, 88, nb_steps, 1)
    idx = list(np.ogrid[[slice(cat_norm.shape[ax]) for ax in range(cat_norm.ndim) if ax != 1]])
    idx.insert(1, argmax)
    cat_norm[tuple(idx)] = 1
    # ----- ----- ----- ----- ----- ----- ----- ----- ----- -----

    if use_binary:
        arr_norm = np.zeros_like(arr)  # (nb_instruments, nb_notes + 1, nb_steps, 1)
        arr_norm[:, :-1] = cat_norm * (1 - activations)
        arr_norm[:, -1:] = activations
        # arr_norm = np.reshape(arr_norm, arr_norm.shape[:-1])        # (nb_instruments, nb_notes + 1, nb_steps)
    else:
        arr_norm = cat_norm

    nb_instruments, nb_notes, nb_steps = arr_norm.shape
    matrix_norm = np.zeros((nb_instruments, nb_notes - 1, nb_steps))
    for instrument in range(nb_instruments):
        duration = 1
        for step in range(nb_steps - 1, -1, -1):
            if arr_norm[instrument, -1, step] == 1:
                duration += 1
            else:
                am = np.argmax(arr_norm[instrument, :, step])
                if am < matrix_norm.shape[1]:
                    matrix_norm[instrument, am, step] = duration
                else:
                    cprint(f'Got an argmax == {am} == size matrix_norm', 'red')
                duration = 1

    return matrix_norm


def converter_func(arr, mono=False, no_duration=False, use_binary=False):
    """

    :param use_binary:
    :param arr:
    :param mono:
    :param no_duration:
    :return:
    """
    if mono:
        arr = converter_func_mono(arr,
                                  use_binary=use_binary)  # Make it consistent      # (nb_instruments, 128, nb_steps)
    else:
        arr = converter_func_poly(arr, no_duration=no_duration)
    return arr


def int_to_note(integer):
    # convert pitch value to the note which is a letter form.
    note_base_name = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave_detector = (integer // 12)
    base_name_detector = (integer % 12)
    note = note_base_name[base_name_detector] + str((int(octave_detector)) - 1)
    if '-' in note:
        note = note_base_name[base_name_detector] + str(0)
        return note
    return note


def matrix_to_midi(matrix, instruments=None, notes_range=None, no_duration=False, mono=False, use_binary=False):
    """

    :param use_binary:
    :param mono:
    :param matrix: shape (nb_instruments, 128, nb_steps, 2)
    :param instruments: The instruments,
    :param notes_range:
    :param no_duration: if True : all notes will be the shortest length possible
    :return:
    """
    nb_instuments, nb_notes, nb_steps = matrix.shape[:3]
    instruments = ['Piano' for _ in range(nb_instuments)] if instruments is None else instruments
    notes_range = (0, 88) if notes_range is None else notes_range

    matrix_norm = converter_func(matrix, mono=mono, no_duration=no_duration, use_binary=use_binary)
    # ---- Delete silence in the beginning of the song ----
    how_many_in_start_zeros = 0
    for step in range(nb_steps):
        one_time_interval = matrix_norm[:, :, step]  # values in a column, (nb_instruments, 128)
        if np.sum(one_time_interval) == 0:
            how_many_in_start_zeros += 1
        else:
            break
    # ---- Delete silence at the end of the song ----
    how_many_in_end_zeros = 0
    for step in range(nb_steps - 1, 0, -1):
        one_time_interval = matrix[:, :, step]  # values in a column
        if np.sum(one_time_interval) == 0:
            how_many_in_end_zeros += 1
        else:
            break
    indice_end = -how_many_in_end_zeros if how_many_in_end_zeros > 0 else nb_steps

    matrix_norm = matrix_norm[:, :,
                  how_many_in_start_zeros: indice_end]  # (nb_instruments, 128, nb_steps_corrected)
    nb_instruments, nb_notes, nb_steps = matrix_norm.shape
    output_notes_instruments = []

    for inst in range(nb_instruments):
        output_notes = []
        for note in range(nb_notes):
            for step in range(nb_steps):
                length_note = matrix_norm[inst, note, step]
                if length_note > 0:  # This is a new note !!
                    new_note = music21.note.Note(pitch=(note + 21 + notes_range[0]),
                                                 duration=music21.duration.Duration(length_note / g.midi.step_per_beat))
                    new_note.offset = step / g.midi.step_per_beat
                    new_note.storedInstrument = midi_inst.string2instrument(instruments[inst])()
                    output_notes.append(new_note)
        output_notes_instruments.append(output_notes)

    return output_notes_instruments


def save_midi(output_notes_list, instruments, path):
    """

    :param output_notes_list: the notes
    :param instruments : le list of the name of the instruments used
    :param path: The .mid file path
    :return:
    """
    print('Converting to Midi ...')
    midi_stream = music21.stream.Stream()
    for i in range(len(instruments)):
        s = music21.stream.Stream()
        for n in output_notes_list[i]:
            s.insert(n.offset, n)
        # p.insert(0, Midi.instrument.Instrument(instrumentName=instruments[i]))
        s.insert(0, midi_inst.string2instrument(instruments[i])())
        s.partName = instruments[i]
        midi_stream.insert(0, s)
    midi_stream.write('Midi', fp=path)
    print(colored(str(path) + ' saved', 'green'))


def print_informations(nb_steps, matrix, notes_list, verbose):
    """

    :param nb_steps:
    :param matrix:
    :param notes_list:
    :param verbose:
    :return:
    """
    nb_instruments = matrix.shape[0]
    nb_notes_seed = [0 for i in range(nb_instruments)]
    nb_notes_generated = [0 for i in range(nb_instruments)]
    for index, instrument_notes in enumerate(notes_list):
        for n in instrument_notes:
            if n.offset <= nb_steps / g.midi.step_per_beat:
                nb_notes_seed[index] += 1
            else:
                nb_notes_generated[index] += 1
    if verbose == 0:
        pass
    elif verbose == 1:
        print('notes in seed :', colored(nb_notes_seed, 'magenta'), 'notes generated :',
              colored(nb_notes_generated, 'magenta'), 'for length', colored(matrix.shape[2], 'magenta'))


def midinote_to_note(midinote, notes_range=(0, 88)):
    """

    :param midinote:
    :param notes_range:
    :return:
    """
    return midinote - (21 + notes_range[0])
