import src.global_variables as g
import numpy as np
import music21
from termcolor import colored

import src.midi.instruments as midi_inst


def normalize_activation(arr, threshold=0.5):
    """

    :param arr: (nb_instruments, nb_steps=1, 88, 2)
    :param threshold:
    :return: the same array but only with one and zeros for the activation part ([:, :, :, 0])
    """
    activations = arr[:, :, :, 0]
    for i in range(0, 20):
        if arr[1, 0, -i, 0] >= threshold:
            print(i, '0.5 <', arr[1, 0, -i, 0])
    np.place(activations, threshold <= activations, 1)
    np.place(activations, activations < threshold, 0)
    arr[:, :, :, 0] = activations
    return arr


def converter_func(arr, no_duration=False):
    """

    :param arr: (nb_instruments, 88, nb_steps, 2)
    :param no_duration: if True : all notes will be the shortest length possible
    :return:
    """
    activations = arr[:, :, :, 0]
    durations = arr[:, :, :, 1]

    np.place(activations, 0.5 <= activations, 1)
    np.place(activations, activations < 0.5, 0)

    durations = np.ceil(durations * g.max_length_note_array)
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


def matrix_to_midi(matrix, instruments=None, notes_range=None, no_duration=False):
    """

    :param matrix: shape (nb_instruments, 128, nb_steps, 2)
    :param instruments: The instruments,
    :param notes_range:
    :param no_duration: if True : all notes will be the shortest length possible
    :return:
    """
    nb_instuments, nb_notes, nb_steps, _ = matrix.shape
    instruments = ['Piano' for _ in range(nb_instuments)] if instruments is None else instruments
    notes_range = (0, 88) if notes_range is None else notes_range

    matrix_norm = converter_func(matrix,
                                 no_duration=no_duration)  # Make it consistent      # (nb_instruments, 128, nb_steps)
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
                                                 duration=music21.duration.Duration(length_note / g.step_per_beat))
                    new_note.offset = step / g.step_per_beat
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
    print('Converting to midi ...')
    midi_stream = music21.stream.Stream()
    for i in range(len(instruments)):
        s = music21.stream.Stream()
        for n in output_notes_list[i]:
            s.insert(n.offset, n)
        # p.insert(0, midi.instrument.Instrument(instrumentName=instruments[i]))
        s.insert(0, midi_inst.string2instrument(instruments[i])())
        s.partName = instruments[i]
        midi_stream.insert(0, s)
    midi_stream.write('midi', fp=path)
    print(colored(path + ' saved', 'green'))
