import src.global_variables as g
import numpy as np
import music21
import functools

import src.midi.instruments as midi_inst


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
        last_offset = max(map(lambda x: int(x), offsets))
    except ValueError:
        print('Value Error')
        return None, None, None

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
        if '.' not in str(note):  # it is not chord. Single note.
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
    if type(duration) is str:
        if '/' in duration:
            numerator = float(duration.split('/')[0])
            denominator = float(duration.split('/')[1])
            duration = str(float(numerator / denominator))
        return duration
    else:
        return str(float(duration))


def midi_to_matrix(filename, instruments, length=None):  # convert midi file to matrix for DL architecture.

    midi = music21.converter.parse(filename)  # Load the file
    parts = music21.instrument.partitionByInstrument(midi)

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
    our_matrixes = []
    # print(instrument_names)
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

            our_matrix = notes_to_matrix(notes, durations, offsets)

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

    final_matrix = np.zeros((len(our_matrixes), 128, max_len))
    for i in range(len(our_matrixes)):
        final_matrix[i, :, :len(our_matrixes[i][0])] = our_matrixes[i]
    return final_matrix
