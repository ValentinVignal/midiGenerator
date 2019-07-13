import src.global_variables as g
import numpy as np
import music21

import src.midi.instruments as midi_inst


def converter_func(arr, first_touch=1.0, continuation=0.0, lower_bound=g.lower_bound, upper_bound=g.upper_bound):
    # I can write this function thanks to https://stackoverflow.com/questions/16343752/numpy-where-function-multiple-conditions
    # first touch represent start for note, continuation represent continuation for first touch, 0 represent end or rest
    np.place(arr, arr < lower_bound, -1.0)
    np.place(arr, (lower_bound <= arr) & (arr < upper_bound), 0.0)
    np.place(arr, arr >= upper_bound, 1.0)
    return arr


def how_many_repetitive_func(array, from_where=0, continuation=0.0):
    new_array = array[from_where:]
    count_repetitive = 1
    for i in new_array:
        if i != continuation:
            return count_repetitive
        else:
            count_repetitive += 1
    return count_repetitive


def int_to_note(integer):
    # convert pitch value to the note which is a letter form.
    note_base_name = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave_detector = (integer // 12)
    base_name_detector = (integer % 12)
    note = note_base_name[base_name_detector] + str((int(octave_detector)) - 1)
    if ('-' in note):
        note = note_base_name[base_name_detector] + str(0)
        return note
    return note


def matrix_to_midi(matrix, random=0):
    first_touch = 1.0
    continuation = 0.0
    y_axis, x_axis = matrix.shape
    output_notes = []
    offset = 0

    # Delete rows until the row which include 'first_touch'
    how_many_in_start_zeros = 0
    for x_axis_num in range(x_axis):
        one_time_interval = matrix[:, x_axis_num]  # values in a column
        one_time_interval_norm = converter_func(one_time_interval)
        if first_touch not in one_time_interval_norm:
            how_many_in_start_zeros += 1
        else:
            break

    how_many_in_end_zeros = 0
    for x_axis_num in range(x_axis - 1, 0, -1):
        one_time_interval = matrix[:, x_axis_num]  # values in a column
        one_time_interval_norm = converter_func(one_time_interval)
        if first_touch not in one_time_interval_norm:
            how_many_in_end_zeros += 1
        else:
            break

    # print('How many rows for non-start note at beginning:', how_many_in_start_zeros)
    # print('How many rows for non-start note at end:', how_many_in_end_zeros)

    matrix = matrix[:, how_many_in_start_zeros:]
    y_axis, x_axis = matrix.shape
    # print('size : {0}, {1}'.format(y_axis, x_axis))

    for y_axis_num in range(y_axis):
        one_freq_interval = matrix[y_axis_num, :]  # bir columndaki değerler
        #  freq_val = 0 # columdaki hangi rowa baktığımızı akılda tutmak için
        one_freq_interval_norm = converter_func(one_freq_interval)
        # print (one_freq_interval)
        i = 0
        offset = 0

        if (random):

            while i < len(one_freq_interval):
                how_many_repetitive = 0
                temp_i = i
                if one_freq_interval_norm[i] == first_touch:
                    how_many_repetitive = how_many_repetitive_func(one_freq_interval_norm, from_where=i + 1,
                                                                   continuation=continuation)
                    i += how_many_repetitive

                if how_many_repetitive > 0:
                    random_num = np.random.randint(3, 6)
                    new_note = music21.note.Note(int_to_note(y_axis_num),
                                                 duration=music21.duration.Duration(
                                                     0.25 * random_num * how_many_repetitive))
                    new_note.offset = 0.25 * temp_i * 2
                    new_note.storedInstrument = music21.instrument.Piano()
                    output_notes.append(new_note)
                else:
                    i += 1


        else:

            while i < len(one_freq_interval):
                how_many_repetitive = 0
                temp_i = i
                if one_freq_interval_norm[i] == first_touch:
                    how_many_repetitive = how_many_repetitive_func(one_freq_interval_norm, from_where=i + 1,
                                                                   continuation=continuation)
                    i += how_many_repetitive

                if how_many_repetitive > 0:
                    new_note = music21.note.Note(int_to_note(y_axis_num),
                                                 duration=music21.duration.Duration(0.25 * how_many_repetitive))
                    new_note.offset = 0.25 * temp_i
                    new_note.storedInstrument = music21.instrument.Piano()
                    output_notes.append(new_note)
                else:
                    i += 1

    return output_notes


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


############################################
###### Valentin #####
############################################


def save_midi(output_notes_list, instruments, path):
    """

    :param output_notes_list: the notes
    :param instruments : le list of the name of the instruments used
    :param path: The .mid file path
    :return:
    """
    midi_stream = music21.stream.Stream()
    for i in range(len(instruments)):
        p = music21.stream.Part()
        p.append(output_notes_list[i])
        # p.insert(0, midi.instrument.Instrument(instrumentName=instruments[i]))
        p.insert(midi_inst.string2instrument(instruments[i])())
        midi_stream.insert(0, p)
    midi_stream.write('midi', fp=path)
    print(path, 'saved')

