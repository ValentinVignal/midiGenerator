import argparse
import os
import pickle
import music21
import numpy as np


def allMidiFiles(path, small_data):
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

def check_float(duration):  #  this function fix the issue which comes from some note's duration.
    # For instance some note has duration like 14/3 or 7/3.
    if ('/' in duration):
        numerator = float(duration.split('/')[0])
        denominator = float(duration.split('/')[1])
        duration = str(float(numerator / denominator))
    return duration

def main():
    """
        Entry point
    """

    parser = argparse.ArgumentParser(description='Program to train a model over a midi dataset')
    parser.add_argument('--epochs', type=int, default=5, metavar='N',
                        help='number of epochs to train (default: 5)')
    parser.add_argument('--lr', type=float, default=0.00001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--pc', action='store_true', default=False,
                        help='to work on a small computer with a cpu')
    parser.add_argument('--seed', type=int, default=1234, metavar='S',
                        help='random seed (default: 1234)')
    parser.add_argument('--log-interval', type=int, default=5, metavar='N',
                        help='how many batch to wait before logging training status')
    parser.add_argument('--model', type=str, default='1',
                        help='The model of the Neural Network used for the interpolation')
    parser.add_argument('--batch', type=int, default=1,
                        help='The number of the batchs')
    parser.add_argument('--gpu', type=str, default='0',
                        help='What GPU to use')

    args = parser.parse_args()

    if args.pc:
        data_path = '../Dataset/lmd_matched'
    else:
        data_path = '../../../../../../storage1/valentin/lmd_matched'

    min_value = 0.00
    lower_first = 0.00

    lower_second = 0.5
    upper_first = 0.5

    upper_second = 1.0
    max_value = 1.0


    data_p = os.path.join(data_path, 'data.p')      # Pickle file with the informations of the data set
    if os.path.exists(data_p):
        with open(data_p, 'rb') as dump_file:
            d = pickle.load(dump_file)
            data_midi = d['midi']
    else:
        data_midi = allMidiFiles(data_path, args.pc)
        with open(data_p, 'wb') as dump_file:
            pickle.dump({
                'midi': data_midi
            }, dump_file)


    filename = data_midi[0]

    midi = music21.converter.parse(filename)
    notes_to_parse = None

    parts = music21.instrument.partitionByInstrument(midi)

    instrument_names = []

    try:
        for instrument in parts:  # Learn names of instruments.
            name = (str(instrument).split(' ')[-1])[:-1]
            instrument_names.append(name)

    except TypeError:
        print('Type is not iterable.')
        return None

    # Just take piano part. For the future works, we can use different instrument.
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
        if isinstance(element, music21.note.Note):  # If it is single note
            notes.append(note_to_int(str(element.pitch)))  # Append note's integer value to "notes" list.
            duration = str(element.duration)[27:-1]
            durations.append(check_float(duration))
            offsets.append(element.offset)

        elif isinstance(element, music21.chord.Chord):  # If it is chord
            notes.append('.'.join(str(note_to_int(str(n)))
                                  for n in element.pitches))
            duration = str(element.duration)[27:-1]
            durations.append(check_float(duration))
            offsets.append(element.offset)

    try:
        last_offset = int(offsets[-1])
    except IndexError:
        print('Index Error')
        return (None, None, None)

    total_offset_axis = last_offset * 4 + (8 * 4)
    our_matrix = np.random.uniform(min_value, lower_first, (128, int(total_offset_axis)))

    for (note, duration, offset) in zip(notes, durations, offsets):
        how_many = int(float(duration) / 0.25)  # indicates time duration for single note.

        # Define difference between single and double note.
        # I have choose the value for first touch, the another value for continuation.
        # Lets make it randomize

        # I choose to use uniform distrubition. Maybe, you can use another distrubition like Gaussian.

        first_touch = np.random.uniform(upper_second, max_value, 1)
        continuation = np.random.uniform(lower_second, upper_first, 1)

        if ('.' not in str(note)):  # It is not chord. Single note.
            our_matrix[note, int(offset * 4)] = first_touch
            our_matrix[note, int((offset * 4) + 1): int((offset * 4) + how_many)] = continuation

        else:  # For chord
            chord_notes_str = [note for note in note.split('.')]
            chord_notes_float = list(map(int, chord_notes_str))  # Take notes in chord one by one

            for chord_note_float in chord_notes_float:
                our_matrix[chord_note_float, int(offset * 4)] = first_touch
                our_matrix[chord_note_float, int((offset * 4) + 1): int((offset * 4) + how_many)] = continuation

    print(our_matrix)


    print('Done')








if __name__ == '__main__':
    # create a separate main function because original main function is too mainstream
    main()
