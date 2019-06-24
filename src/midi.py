import src.global_variables as g
import numpy as np
import music21


def note_to_int(note):  # converts the note's letter to pitch value which is integer form.
    # source: https://musescore.org/en/plugin-development/note-pitch-values
    # idea: https://github.com/bspaans/python-mingus/blob/master/mingus/core/notes.py
    note_base_name = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    if ('#-' in note):
        first_letter = note[0]
        base_value = note_base_name.index(first_letter)
        octave = note[3]
        value = base_value + 12 * (int(octave) - (-1))

    elif ('#' in note):  # not totally sure, source: http://www.pianofinders.com/educational/WhatToCallTheKeys1.htm
        first_letter = note[0]
        base_value = note_base_name.index(first_letter)
        octave = note[2]
        value = base_value + 12 * (int(octave) - (-1))

    elif ('-' in note):
        first_letter = note[0]
        base_value = note_base_name.index(first_letter)
        octave = note[2]
        value = base_value + 12 * (int(octave) - (-1))

    else:
        first_letter = note[0]
        base_val = note_base_name.index(first_letter)
        octave = note[1]
        value = base_val + 12 * (int(octave) - (-1))

    return value


def notes_to_matrix(notes, durations, offsets, min_value=g.min_value, lower_first=g.lower_first,
                    lower_second=g.lower_second,
                    upper_first=g.upper_first, upper_second=g.upper_second,
                    max_value=g.max_value):
    # I want to represent my notes in matrix form. X axis will represent time, Y axis will represent pitch values.
    # I should normalize my matrix between 0 and 1.
    # So that I will represent rest with (min_value, lower_first), continuation with [lower_second, upper_first]
    # and first touch with (upper_second, max_value)
    # First touch means that you press the note and it cause to 1 time duration playing. Continuation
    # represent the continuum of this note playing.

    try:
        last_offset = int(offsets[-1])
    except IndexError:
        print('Index Error')
        return (None, None, None)

    total_offset_axis = last_offset * 4 + (8 * 4)
    our_matrix = np.random.uniform(min_value, lower_first, (128, int(total_offset_axis)))
    # creates matrix and fills with (-1, -0.3), this values will represent the rest.

    for (note, duration, offset) in zip(notes, durations, offsets):
        how_many = int(float(duration) / 0.25)  # indicates time duration for single note.

        # Define difference between single and double note.
        # I have choose the value for first touch, the another value for contiunation
        # lets make it randomize
        first_touch = np.random.uniform(upper_second, max_value, 1)
        # continuation = np.random.randint(low=-1, high=1) * np.random.uniform(lower_second, upper_first, 1)
        continuation = np.random.uniform(lower_second, upper_first, 1)
        if ('.' not in str(note)):  # it is not chord. Single note.
            our_matrix[note, int(offset * 4)] = first_touch
            our_matrix[note, int((offset * 4) + 1): int((offset * 4) + how_many)] = continuation

        else:  # For chord
            chord_notes_str = [note for note in note.split('.')]
            chord_notes_float = list(map(int, chord_notes_str))  # take notes in chord one by one

            for chord_note_float in chord_notes_float:
                our_matrix[chord_note_float, int(offset * 4)] = first_touch
                our_matrix[chord_note_float, int((offset * 4) + 1): int((offset * 4) + how_many)] = continuation

    return our_matrix


def check_float(duration):  #  this function fix the issue which comes from some note's duration.
    # For instance some note has duration like 14/3 or 7/3.
    if ('/' in duration):
        numerator = float(duration.split('/')[0])
        denominator = float(duration.split('/')[1])
        duration = str(float(numerator / denominator))
    return duration


def midi_to_matrix(filename, length=250):  # convert midi file to matrix for DL architecture.

    midi = music21.converter.parse(filename)
    notes_to_parse = None

    parts = music21.instrument.partitionByInstrument(midi)

    instrument_names = []

    try:
        for instrument in parts:  # learn names of instruments
            name = (str(instrument).split(' ')[-1])[:-1]
            instrument_names.append(name)

    except TypeError:
        print('Type is not iterable.')
        return None

    # just take piano part
    try:
        piano_index = instrument_names.index('Piano')
    except ValueError:
        print('%s have not any Piano part' % (filename))
        return None

    notes_to_parse = parts.parts[piano_index].recurse()

    duration_piano = float(check_float((str(notes_to_parse._getDuration()).split(' ')[-1])[:-1]))

    durations = []
    notes = []
    offsets = []

    for element in notes_to_parse:
        if isinstance(element, music21.note.Note):  # if it is single note
            notes.append(note_to_int(str(element.pitch)))
            duration = str(element.duration)[27:-1]
            durations.append(check_float(duration))
            offsets.append(element.offset)

        elif isinstance(element, music21.chord.Chord):  # if it is chord
            notes.append('.'.join(str(note_to_int(str(n)))
                                  for n in element.pitches))
            duration = str(element.duration)[27:-1]
            durations.append(check_float(duration))
            offsets.append(element.offset)

    our_matrix = notes_to_matrix(notes, durations, offsets)

    try:
        freq, time = our_matrix.shape
    except AttributeError:
        print("'tuple' object has no attribute 'shape'")
        return None

    if (time >= length):
        return (
            our_matrix[:, :length])  #  We have to set all individual note matrix to same shape for Generative DL.
    else:
        print('%s have not enough duration' % (filename))


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
        if (i != continuation):
            return (count_repetitive)
        else:
            count_repetitive += 1
    return (count_repetitive)


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
        if (first_touch not in one_time_interval_norm):
            how_many_in_start_zeros += 1
        else:
            break

    how_many_in_end_zeros = 0
    for x_axis_num in range(x_axis - 1, 0, -1):
        one_time_interval = matrix[:, x_axis_num]  # values in a column
        one_time_interval_norm = converter_func(one_time_interval)
        if (first_touch not in one_time_interval_norm):
            how_many_in_end_zeros += 1
        else:
            break

    print('How many rows for non-start note at beginning:', how_many_in_start_zeros)
    print('How many rows for non-start note at end:', how_many_in_end_zeros)

    matrix = matrix[:, how_many_in_start_zeros:]
    y_axis, x_axis = matrix.shape
    print(y_axis, x_axis)

    for y_axis_num in range(y_axis):
        one_freq_interval = matrix[y_axis_num, :]  # bir columndaki değerler
        #  freq_val = 0 # columdaki hangi rowa baktığımızı akılda tutmak için
        one_freq_interval_norm = converter_func(one_freq_interval)
        # print (one_freq_interval)
        i = 0
        offset = 0

        if (random):

            while (i < len(one_freq_interval)):
                how_many_repetitive = 0
                temp_i = i
                if (one_freq_interval_norm[i] == first_touch):
                    how_many_repetitive = how_many_repetitive_func(one_freq_interval_norm, from_where=i + 1,
                                                                   continuation=continuation)
                    i += how_many_repetitive

                if (how_many_repetitive > 0):
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

            while (i < len(one_freq_interval)):
                how_many_repetitive = 0
                temp_i = i
                if (one_freq_interval_norm[i] == first_touch):
                    how_many_repetitive = how_many_repetitive_func(one_freq_interval_norm, from_where=i + 1,
                                                                   continuation=continuation)
                    i += how_many_repetitive

                if (how_many_repetitive > 0):
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

    preds[0:48] = 0  # eliminate notes with low octaves
    preds[100:] = 0  # eliminate notes with very high octaves

    ind = np.argpartition(preds, -1 * num_of_top)[-1 * num_of_top:]
    top_indices_sorted = ind[np.argsort(preds[ind])]

    array = np.random.uniform(0.0, 0.0, (128))
    array[top_indices_sorted[0:num_of_first]] = 1.0
    array[top_indices_sorted[num_of_first:num_of_first + 3]] = 0.5

    return array


