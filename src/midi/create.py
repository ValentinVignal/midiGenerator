import src.global_variables as g
import numpy as np
import music21
import progressbar
import functools

import src.midi.instruments as midi_inst


def converter_func(arr):
    """

    :param arr: (nb_instruments, 128, nb_steps, 2)
    :return:
    """
    activations = arr[:, :, :, 0]
    durations = arr[:, :, :, 1]

    np.place(activations, 0.5 <= activations, 1)
    np.place(activations, activations < 0.5, 0)

    durations = np.ceil(durations * g.max_length_note)
    durations = np.maximum(durations, 1)

    return np.multiply(activations, durations)


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


def matrix_to_midi(matrix, instruments=None):
    """

    :param matrix: shape (nb_instruments, 128, nb_steps)
    :param instruments: The instruments
    :return:
    """
    nb_instuments, nb_notes, nb_steps, _ = matrix.shape
    instruments = ['Piano' for _ in range(nb_instuments)] if instruments is None else instruments

    matrix_norm = converter_func(matrix)  # Make it consistent      # (nb_instruments, 128, nb_steps)
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
                if length_note > 0:    # This is a new note !!
                    new_note = music21.note.Note(int_to_note(note),
                                                 duration=music21.duration.Duration(0.25 * length_note))
                    new_note.offset = 0.25 * step
                    new_note.storedInstrument = midi_inst.string2instrument(instruments[inst])()
                    output_notes.append(new_note)
        output_notes_instruments.append(output_notes)

    return output_notes_instruments


def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)

    num_of_top = 15
    num_of_first = np.random.randint(1, 3)

    # preds[0:48] = 0  # eliminate notes with low octaves
    # preds[100:] = 0  # eliminate notes with very high octaves

    ind = np.argpartition(preds, -1 * num_of_top)[-1 * num_of_top:]
    top_indices_sorted = ind[np.argsort(preds[ind])]  # 15 biggest number

    array = np.random.uniform(0.0, 0.0, (128))
    array[top_indices_sorted[0:num_of_first]] = 1.0
    array[top_indices_sorted[num_of_first:num_of_first + 3]] = 0.5

    return array


#       ############################################
#       ################ Valentin ##################
#       ############################################


def save_midi(output_notes_list, instruments, path):
    """

    :param output_notes_list: the notes
    :param instruments : le list of the name of the instruments used
    :param path: The .mid file path
    :return:
    """
    print('Converting to midi ...')
    nb_notes = functools.reduce(lambda x, y: x + len(y), output_notes_list, 0)
    bar = progressbar.ProgressBar(maxval=nb_notes,
                                  widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage(), ' ',
                                           progressbar.ETA()])
    bar.start()  # To see it working
    e = 0

    midi_stream = music21.stream.Stream()
    for i in range(len(instruments)):
        s = music21.stream.Stream()
        for n in output_notes_list[i]:
            s.insert(n.offset, n)
            e += 1
            bar.update(e)
        # p.insert(0, midi.instrument.Instrument(instrumentName=instruments[i]))
        s.insert(0, midi_inst.string2instrument(instruments[i])())
        s.partName = instruments[i]
        midi_stream.insert(0, s)
    midi_stream.write('midi', fp=path)
    bar.finish()
    print(path, 'saved')
