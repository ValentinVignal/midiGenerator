import src.global_variables as g
import numpy as np
import music21
import functools

import src.midi.instruments as midi_inst


def matrix_rest(dims, rest=g.rest):
    """

    :param dims: Dimension of the matrix
    :param rest: Value to code rest in the matrix (=-1)
    :return: A np array of shape dims and with every element with the value rest
    """
    return rest * np.ones(dims)


def notes_to_matrix_val(notes, durations, offsets, first_touch=g.first_touch, continuation=g.contination, rest=g.rest):
    """

    :param notes: The notes
    :param durations: The duration of the notes
    :param offsets: The offset of the notes
    :param first_touch: Value to code first touch in the array (=1)
    :param continuation: Value to code the continuation of a note in the array (=0)
    :param rest: Value to code the rest in the array (=-1)
    :return: The matrix corresponding to the notes
    """
    try:
        last_offset = max(map(lambda x: int(x), offsets))
    except ValueError:
        print('Value Error')
        return None, None, None
    total_offset_axis = last_offset * 4 + (
            8 * 4)  # nb times * 4 because quarter note + 2 measures (max length of a note)
    our_matrix = matrix_rest((128, int(total_offset_axis)))  # (128, nb_times)

    for (note, duration, offset) in zip(notes, durations, offsets):
        how_many = int(float(duration) / 0.25)  # indicates time duration for single note.
        start = int(offset * 4)
        start_c = int((offset * 4) + 1)
        end = int((offset * 4 + how_many))
        if '.' not in str(note):  # it is not chord. Single note.
            our_matrix[note, start] = first_touch
            if start_c < end:   # Continuation is needed to be coded
                our_matrix[note, start_c: end] = np.maximum(
                    our_matrix[note, start_c: end],
                    np.array(continuation))  # np maximum to keep information first touch if some notes cover themselves

        else:  # For chord
            chord_notes_str = [note for note in note.split('.')]
            chord_notes_float = list(map(int, chord_notes_str))  # take notes in chord one by one

            for chord_note_float in chord_notes_float:
                our_matrix[chord_note_float, start] = first_touch
                if start_c < end:       # continuation is needed to be coded
                    our_matrix[chord_note_float, start_c: end] = np.maximum(
                        our_matrix[chord_notes_float, start_c: end],
                        np.array(continuation))  # np maximum to keep information first touch if some notes cover themselves

    return our_matrix


def check_float(duration):
    """
    This function fix the issue which comes from some note's duration.
    For instance some note has duration like 14/3 or 7/3.
    :param duration:
    :return: array of the coded notes with shape (nb_instruments, 128, length_song)
    """
    if type(duration) is str:
        if '/' in duration:
            numerator = float(duration.split('/')[0])
            denominator = float(duration.split('/')[1])
            duration = str(float(numerator / denominator))
        return duration
    else:
        return str(float(duration))


def midi_to_matrix(filename, instruments, length=None):
    """
    convert midi file to matrix for DL architecture.
    :param filename: path to the midi file
    :param instruments: instruments to train on
    :param length: length max of the song
    :return: matrix with shape
    """
    midi = music21.converter.parse(filename)  # Load the file
    parts = music21.instrument.partitionByInstrument(midi)

    # --- Get the instruments names in the file ---
    instrument_names = []  # Name of the instruments in the file
    try:
        for instrument in parts:
            # learn names of instruments
            name = instrument.partName
            # name = (str(instrument).split(' ')[-1])[
            #        :-1]  # str(instrument) = "<midi.stream.Part object Electric Bass>"
            instrument_names.append(name)
    except TypeError:
        print('Type is not iterable.')
        return None

    # just take instruments desired parts
    our_matrixes = []  # Final matrix for deep learning
    for instrument in instruments:
        similar_instruments = midi_inst.return_similar_instruments(instrument)
        at_least_one = functools.reduce(lambda x, y: (x or (y in instrument_names)), similar_instruments, False)
        if not at_least_one:
            print('{0} have not any {1} part'.format(filename, instrument))
            return None
        else:  # We know there is a similar instrument in it
            notes_to_parse = None
            for similar_instrument in similar_instruments:
                if similar_instrument in instrument_names:
                    instrument_index = instrument_names.index(similar_instrument)
                    if notes_to_parse is None:
                        notes_to_parse = parts.parts[instrument_index]
                    else:
                        notes_to_parse.append(parts.parts[instrument_index])
            notes_to_parse = notes_to_parse.recurse()
            duration = float(check_float(notes_to_parse._getDuration().quarterLength))

            durations = []
            notes = []
            offsets = []

            for element in notes_to_parse:
                if isinstance(element, music21.note.Note):  # if it is single note
                    notes.append(int(element.pitch.midi))  # The code number for the pitch
                    duration = str(element.duration)[27:-1]
                    durations.append(check_float(duration))
                    offsets.append(element.offset)

                elif isinstance(element, music21.chord.Chord):  # if it is chord
                    notes.append('.'.join(str(n.midi)
                                          for n in element.pitches))
                    duration = str(element.duration)[27:-1]
                    durations.append(check_float(duration))
                    offsets.append(element.offset)

            our_matrix = notes_to_matrix_val(notes, durations, offsets)

            try:
                freq, time = our_matrix.shape
            except AttributeError:
                print("'tuple' object has no attribute 'shape'")
                return None

            # To change shape
            if length is not None:
                try:
                    our_matrix = our_matrix[:, :length]
                except IndexError:
                    print('{0} is not long enough, shape : {1}'.format(filename, our_matrix.shape))

            our_matrixes.append(our_matrix)

    # Normalization of the duration : make them all finish at the same time
    max_len = 0
    for matrix in our_matrixes:
        if len(matrix[0]) > max_len:  # matrix has shape : (128, length)
            max_len = len(matrix[0])

    final_matrix = matrix_rest((len(our_matrixes), 128, max_len))   # (nb_instruments, 128, max_len)
    for i in range(len(our_matrixes)):
        final_matrix[i, :, :len(our_matrixes[i][0])] = our_matrixes[i]
    return final_matrix
