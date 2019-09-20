import src.global_variables as g
import numpy as np
import music21
import functools
from termcolor import colored, cprint
import math
import os

import src.midi.instruments as midi_inst


def no_silence(matrix):
    """
    Erase silence at the begining and at the end of the matrix
    :param matrix: np array (nb_instruments, input_size, nb_steps, 2)
    :return:
    """
    start, end = 0, matrix.shape[2]
    while start < end and np.all(matrix[:, :, start, 0] == 0):
        start += 1
    while start < end and np.all(matrix[:, :, end - 1, 0] == 0):
        end -= 1
    # To have all the mesures
    start = math.floor(start / (4 * g.step_per_beat)) * 4 * g.step_per_beat
    end = math.ceil(end / (4 * g.step_per_beat)) * 4 * g.step_per_beat + 1
    return matrix[:, :, start:end, :]  # (nb_instruments, input_size, nb_steps, 2)


def notes_to_matrix(notes, durations, offsets):
    """

    :param notes: The notes
    :param durations: The duration of the notes
    :param offsets: The offset of the notes
    :return: The matrix corresponding to the notes
    """
    try:
        last_offset = max(map(lambda x: int(x), offsets))
    except ValueError:
        print(colored('Value Error', 'red'))
        return None, None, None
    total_offset_axis = last_offset * 4 + (
            8 * 4)  # nb times * 4 because quarter note + 2 measures (max length of a note)
    our_matrix = np.zeros(
        (128, math.ceil(total_offset_axis / (4 * g.step_per_beat)) * 4 * g.step_per_beat, 2))  # (128, nb_times, 2)

    for (note, duration, offset) in zip(notes, durations, offsets):
        # how_many = int(float(duration) / 0.25)  # indicates time duration for single note.
        start = int(offset * 4)
        if '.' not in str(note):  # it is not chord. Single note.
            our_matrix[note, start, 0] = 1
            our_matrix[note, start, 1] = float(duration) / g.max_length_note_music21

        else:  # For chord
            chord_notes_str = [note for note in note.split('.')]
            chord_notes_float = list(map(int, chord_notes_str))  # take notes in chord one by one

            for chord_note_float in chord_notes_float:
                our_matrix[chord_note_float, start, 0] = 1
                our_matrix[chord_note_float, start, 1] = float(duration) / g.max_length_note_music21

    # our_matrix is (128, nb_steps, 2)
    return our_matrix[21:109]  # From A0 to C8 (88, nb_steps, 2)


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


def midifile_to_stream(filename, keep_drums=False):
    """

    :param filename:
    :param keep_drums:
    :return:
    """
    mf = music21.midi.MidiFile()
    mf.open(filename)
    mf.read()
    mf.close()
    # Look for the track with percussion (always channel 10)
    has_drums = False
    drums_tracks_indexes = []
    for t in range(len(mf.tracks)):
        for c in mf.tracks[t].getChannels():
            if c == 10:
                has_drums = True
                drums_tracks_indexes.append(t)
    drums_tracks = []
    for dti in sorted(drums_tracks_indexes, reverse=True):
        drums_tracks.append(mf.tracks.pop(dti))
    # Now there is no percussions in mf

    if keep_drums and has_drums:
        raise NotImplementedError("Can't keep drums for now")
    try:
        return music21.midi.translate.midiFileToStream(mf)
    except IndexError:
        if has_drums and not keep_drums:
            print(colored('Only drums in file {0}'.format(filename), 'red'))
        else:
            print(colored('File is empty {0}'.format(filename), 'red'))
        return None
    except music21.exceptions21.StreamException:
        print(colored('There is no tracks in file {0}'.format(filename), 'red'))
        return None


def midi_to_matrix(filename, instruments, length=None, print_instruments=False, notes_range=None):
    """
    convert midi file to matrix for DL architecture.
    :param filename: path to the midi file
    :param instruments: instruments to train on
    :param length: length max of the song
    :param print_instruments: bool
    :return: matrix with shape
    """
    midi = midifile_to_stream(filename)  # Load the file
    if midi is None:
        return None
    parts = music21.instrument.partitionByInstrument(midi)
    notes_range = (0, 88) if notes_range is None else notes_range

    # --- Get the instruments names in the file ---
    instrument_names = []  # Name of the instruments in the file
    try:
        for instrument in parts:
            # learn names of instruments
            name = instrument.partName
            if name is None:
                if len(instrument_names) > 0:
                    name = instrument_names[-1]
                else:
                    name = 'Piano'
            # name = (str(instrument).split(' ')[-1])[
            #        :-1]  # str(instrument) = "<midi.stream.Part object Electric Bass>"
            instrument_names.append(name)
        if print_instruments:
            print('instruments :', instrument_names)
    except TypeError:
        print(colored('Type is not iterable.', 'red'))
        return None

    # just take instruments desired parts
    our_matrixes = []  # Final matrix for deep learning
    for instrument in instruments:
        similar_instruments = midi_inst.return_similar_instruments(instrument)
        at_least_one = functools.reduce(lambda x, y: (x or (y in instrument_names)), similar_instruments, False)
        if not at_least_one:
            print(colored('{0} doesn t have any {1} part'.format(filename, instrument), 'red'))

            return None
        else:  # We know there is a similar instrument in it
            notes_to_parse = music21.stream.Stream()
            for similar_instrument in similar_instruments:
                for instrument_index, instrument_name in enumerate(instrument_names):
                    if similar_instrument == instrument_name:
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
                elif isinstance(element, music21.meter.TimeSignature):
                    # Check if it is correct
                    if element.ratioString != '4/4':
                        cprint('Unwanted time signature : {0}'.format(element.ratioString), 'red')
                        return None
            our_matrix = notes_to_matrix(notes, durations, offsets)  # (88, nb_steps, 2)

            try:
                freq, time, _ = our_matrix.shape
            except AttributeError:
                print(colored("'tuple' object has no attribute 'shape'", 'red'))
                return None

            # To change shape
            if length is not None:
                try:
                    our_matrix = our_matrix[:, :length, :]
                except IndexError:
                    print(colored('{0} is not long enough, shape : {1}'.format(filename, our_matrix.shape), 'red'))

            our_matrixes.append(our_matrix[notes_range[0]:notes_range[1]])

    # Normalization of the duration : make them all finish at the same time
    max_len = 0
    for matrix in our_matrixes:
        if len(matrix[0]) > max_len:  # matrix has shape : (88, length, 2)
            max_len = len(matrix[0])

    final_matrix = np.zeros(
        (len(our_matrixes), notes_range[1] - notes_range[0], max_len, 2))  # (nb_instruments, 88, max_len, 2)
    for i in range(len(our_matrixes)):
        final_matrix[i, :, :len(our_matrixes[i][0]), :] = our_matrixes[i]
    final_matrix = no_silence(final_matrix)
    return final_matrix


def midi_to_matrix_bach(filename, length=None, print_instruments=False, notes_range=None):
    """
    convert midi file to matrix for DL architecture.
    :param filename: path to the midi file
    :param length: length max of the song
    :param print_instruments: bool
    :return: matrix with shape
    """
    midi = midifile_to_stream(filename)  # Load the file
    if midi is None:
        return None
    if len(midi) != 4:
        cprint('Wrong number of parts : {0}'.format(len(midi)), 'red')
        return None
    parts = midi        # No partition by Instruments
    notes_range = (0, 88) if notes_range is None else notes_range

    # --- Get the instruments names in the file ---
    instrument_names = []  # Name of the instruments in the file
    try:
        for instrument in parts:
            # learn names of instruments
            name = instrument.partName
            if name is None:
                if len(instrument_names) > 0:
                    name = instrument_names[-1]
                else:
                    name = 'Piano'
            # name = (str(instrument).split(' ')[-1])[
            #        :-1]  # str(instrument) = "<midi.stream.Part object Electric Bass>"
            instrument_names.append(name)
        if print_instruments:
            print('instruments :', instrument_names)
    except TypeError:
        print(colored('Type is not iterable.', 'red'))
        return None

    # just take instruments desired parts
    our_matrixes = []  # Final matrix for deep learning
    all_first_offset = []
    for instrument in range(4):
        try:
            notes_to_parse = parts.parts[instrument]
        except TypeError:
            print(colored('Type is not iterable.', 'red'))
            return None

        notes_to_parse = notes_to_parse.recurse()
        duration = float(check_float(notes_to_parse._getDuration().quarterLength))

        durations = []
        notes = []
        offsets = []

        in_anacrouse = False
        anacrouse_value = None
        first_offset = 0

        for element in notes_to_parse:
            if in_anacrouse and element.offset > eval(anacrouse_value) * 4 + 2:
                cprint('Unwanted time signature : {0}'.format(anacrouse_value), 'red')
                return None
            if isinstance(element, music21.note.Note) and not in_anacrouse:  # if it is single note
                notes.append(int(element.pitch.midi))  # The code number for the pitch
                duration = str(element.duration)[27:-1]
                durations.append(check_float(duration))
                offsets.append(element.offset)

            elif isinstance(element, music21.chord.Chord) and not in_anacrouse:  # if it is chord
                notes.append('.'.join(str(n.midi)
                                      for n in element.pitches))
                duration = str(element.duration)[27:-1]
                durations.append(check_float(duration))
                offsets.append(element.offset)
            elif isinstance(element, music21.meter.TimeSignature):
                # Check if it is correct
                if element.ratioString != '4/4':
                    if len(notes) == 0:     # Then it is an anacrouse
                        in_anacrouse = True
                        anacrouse_value = element.ratioString
                    else:
                        cprint('Unwanted time signature : {0}'.format(element.ratioString), 'red')
                        return None
                else:
                    in_anacrouse = False
                    first_offset = element.offset
        our_matrix = notes_to_matrix(notes, durations, offsets)  # (88, nb_steps, 2)
        if first_offset != 0:
            all_first_offset.append(first_offset)

        try:
            freq, time, _ = our_matrix.shape
        except AttributeError:
            print(colored("'tuple' object has no attribute 'shape'", 'red'))
            return None

        # To change shape
        if length is not None:
            try:
                our_matrix = our_matrix[:, :length, :]
            except IndexError:
                print(colored('{0} is not long enough, shape : {1}'.format(filename, our_matrix.shape), 'red'))

        our_matrixes.append(our_matrix[notes_range[0]:notes_range[1]])

    # Normalization of the duration : make them all finish at the same time
    max_len = 0
    for matrix in our_matrixes:
        if len(matrix[0]) > max_len:  # matrix has shape : (88, length, 2)
            max_len = len(matrix[0])

    final_matrix = np.zeros(
        (len(our_matrixes), notes_range[1] - notes_range[0], max_len, 2))  # (nb_instruments, 88, max_len, 2)
    for i in range(len(our_matrixes)):
        final_matrix[i, :, :len(our_matrixes[i][0]), :] = our_matrixes[i]

    if len(all_first_offset) >= 1:
        if all(f_o == all_first_offset[0] for f_o in all_first_offset):
            final_matrix = final_matrix[:, :, int(all_first_offset[0] * g.step_per_beat):, :]
        else:
            cprint('Anacrouses not with the same length :{0}'.format(all_first_offset), 'red')

    final_matrix = no_silence(final_matrix)
    return final_matrix


def all_midi_files(path, small_data):
    """

    :param path: the root path
    :param small_data: if we want to keep only a small amount of data
    :return: An array of all the path of all the .mid files in the directory
    """
    nb_small_data = 10
    fichiers = []
    if small_data:
        j = 0
        for root, dirs, files in os.walk(path):
            if j == nb_small_data:
                break
            for i in files:
                if j == nb_small_data:
                    break
                if i.endswith('.mid'):
                    fichiers.append(os.path.join(root, i))
                    j += 1
    else:
        for root, dirs, files in os.walk(path):
            for i in files:
                if i.endswith('.mid'):
                    fichiers.append(os.path.join(root, i))

    return fichiers

