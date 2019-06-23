import argparse
import os
import pickle
import music21
import numpy as np
from pathlib import Path
import tensorflow as tf
import random
import bottleneck



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
    data_transformed_path = data_path + '_transformed'
    if not os.path.exists(data_transformed_path):
        os.mkdir(data_transformed_path)

    min_value = 0.00
    lower_first = 0.00

    lower_second = 0.5
    upper_first = 0.5

    upper_second = 1.0
    max_value = 1.0


    data_p = os.path.join(data_transformed_path, 'data.p')      # Pickle file with the informations of the data set
    if os.path.exists(data_p):
        with open(data_p, 'rb') as dump_file:
            d = pickle.load(dump_file)
            all_midi_paths = d['midi']
    else:
        all_midi_paths = allMidiFiles(data_path, args.pc)
        with open(data_p, 'wb') as dump_file:
            pickle.dump({
                'midi': all_midi_paths
            }, dump_file)

    def notes_to_matrix(notes, durations, offsets, min_value=min_value, lower_first=lower_first,
                        lower_second=lower_second,
                        upper_first=upper_first, upper_second=upper_second,
                        max_value=max_value):

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

    ##################################
    ##################################
    ##################################

    midis_array_path = os.path.join(data_transformed_path, 'midis_array.npy')
    pl_midis_array_path = Path(midis_array_path)

    if pl_midis_array_path.is_file():
        midis_array = np.load(midis_array_path)

    else:
        print(os.getcwd())
        print(all_midi_paths)
        matrix_of_all_midis = []

        # All midi have to be in same shape.

        for single_midi_path in all_midi_paths:
            print(single_midi_path)
            matrix_of_single_midi = midi_to_matrix(single_midi_path, length=250)
            if (matrix_of_single_midi is not None):
                matrix_of_all_midis.append(matrix_of_single_midi)
                print(matrix_of_single_midi.shape)
        midis_array = np.asarray(matrix_of_all_midis)
        midis_array = np.transpose(midis_array, (0, 2, 1))
        np.save(midis_array_path, midis_array)

    midis_array = np.reshape(midis_array, (-1, 128))

    print('midis_array : {0}'.format(midis_array.shape))

    max_len = 18  # how many column will take account to predict next column.
    step = 1  # step size.

    previous_full = []
    predicted_full = []

    for i in range(0, midis_array.shape[0] - max_len, step):
        prev = midis_array[i:i + max_len, ...]  # take max_len column.
        pred = midis_array[i + max_len, ...]  # take (max_len)th column.
        previous_full.append(prev)
        predicted_full.append(pred)

    previous_full = np.asarray(previous_full).astype('float64')
    predicted_full = np.asarray(predicted_full).astype('float64')

    print(previous_full.shape)
    print(predicted_full.shape)

    ###################################
    ###################################
    ###################################

    print('Definition of the graph ...')

    midi_shape = (max_len, 128)

    input_midi = tf.keras.Input(midi_shape)

    x = tf.keras.layers.LSTM(1024, return_sequences=True, unit_forget_bias=True)(input_midi)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    # compute importance for each step
    attention = tf.keras.layers.Dense(1, activation='tanh')(x)
    attention = tf.keras.layers.Flatten()(attention)
    attention = tf.keras.layers.Activation('softmax')(attention)
    attention = tf.keras.layers.RepeatVector(1024)(attention)
    attention = tf.keras.layers.Permute([2, 1])(attention)

    multiplied = tf.keras.layers.Multiply()([x, attention])
    sent_representation = tf.keras.layers.Dense(512)(multiplied)

    x = tf.keras.layers.Dense(512)(sent_representation)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.22)(x)

    x = tf.keras.layers.LSTM(512, return_sequences=True, unit_forget_bias=True)(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.22)(x)

    # compute importance for each step
    attention = tf.keras.layers.Dense(1, activation='tanh')(x)
    attention = tf.keras.layers.Flatten()(attention)
    attention = tf.keras.layers.Activation('softmax')(attention)
    attention = tf.keras.layers.RepeatVector(512)(attention)
    attention = tf.keras.layers.Permute([2, 1])(attention)

    multiplied = tf.keras.layers.Multiply()([x, attention])
    sent_representation = tf.keras.layers.Dense(256)(multiplied)

    x = tf.keras.layers.Dense(256)(sent_representation)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.22)(x)

    x = tf.keras.layers.LSTM(128, unit_forget_bias=True)(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.22)(x)

    x = tf.keras.layers.Dense(128, activation='softmax')(x)

    model = tf.keras.Model(input_midi, x)

    ############################

    optimizer = tf.keras.optimizers.SGD(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    #######################################
    #######################################
    #######################################

    print('Training ...')

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

    lower_bound = (lower_first + lower_second) / 2
    upper_bound = (upper_first + upper_second) / 2

    def converter_func(arr, first_touch=1.0, continuation=0.0, lower_bound=lower_bound, upper_bound=upper_bound):
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

    # PAY ATTENTION. From matrix form to midi form, I have to indicate first touch, continuation and rest with unique numbers.
    # I choose -1.0 for rest , 0 for continuation and 1 for first touch.

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
                                             duration=music21.duration.Duration(0.25 * random_num * how_many_repetitive))
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



    epoch_total = 81
    batch_size = 2

    for epoch in range(1, epoch_total):
        print('Epoch:', epoch)
        model.fit(previous_full, predicted_full, batch_size=batch_size, epochs=1,
                  shuffle=True)

        start_index = random.randint(0, len(midis_array) - max_len - 1)

        generated_midi = midis_array[start_index: start_index + max_len]

        if ((epoch % 10) == 0):
            model.save_weights('my_model_weights.h5')

            for temperature in [1.2]:
                print('------ temperature:', temperature)

                for i in range(480):
                    samples = generated_midi[i:]
                    expanded_samples = np.expand_dims(samples, axis=0)
                    preds = model.predict(expanded_samples, verbose=0)[0]
                    preds = np.asarray(preds).astype('float64')

                    next_array = sample(preds, temperature)

                    midi_list = []
                    midi_list.append(generated_midi)
                    midi_list.append(next_array)
                    generated_midi = np.vstack(midi_list)

                generated_midi_final = np.transpose(generated_midi, (1, 0))
                output_notes = matrix_to_midi(generated_midi_final, random=0)
                midi_stream = music21.stream.Stream(output_notes)
                midi_stream.write('midi', fp='lstm_output_v1_{}_{}.mid'.format(epoch, temperature))

    for layer in model.layers:
        lstm_weights = layer.get_weights()  # list of numpy arrays

        print('Lstm weights:', lstm_weights)

    #################################
    ####################################### Generation
    #######################################

    print('Generation...')

    start_index = random.randint(0, len(midis_array) - max_len - 1)

    generated_midi = midis_array[start_index: start_index + max_len]

    for temperature in [0.7, 2.7]:
        print('------ temperature:', temperature)
        generated_midi = midis_array[start_index: start_index + max_len]
        for i in range(680):
            samples = generated_midi[i:]
            expanded_samples = np.expand_dims(samples, axis=0)
            preds = model.predict(expanded_samples, verbose=0)[0]
            preds = np.asarray(preds).astype('float64')

            next_array = sample(preds, temperature)

            midi_list = []
            midi_list.append(generated_midi)
            midi_list.append(next_array)
            generated_midi = np.vstack(midi_list)

        generated_midi_final = np.transpose(generated_midi, (1, 0))
        output_notes = matrix_to_midi(generated_midi_final, random=1)
        midi_stream = music21.stream.Stream(output_notes)
        midi_file_name = ('lstm_out_{}.mid'.format(temperature))
        midi_stream.write('midi', fp=midi_file_name)
        parsed = music21.converter.parse(midi_file_name)
        for part in parsed.parts:
            part.insert(0, music21.instrument.Piano())
        parsed.write('midi', fp=midi_file_name)

    # To see values

    z = -bottleneck.partition(-preds, 20)[:20]
    print(z)
    print('max:', np.max(preds))

    model.save('my_model.h5')
    model.save_weights('my_model_weights.h5')

    print('Done')








if __name__ == '__main__':
    # create a separate main function because original main function is too mainstream
    main()
