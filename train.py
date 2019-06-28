import argparse
import os
import pickle
import music21
import numpy as np
from pathlib import Path
import tensorflow as tf
import random
import bottleneck

import src.midi as midi



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



def main():
    """
        Entry point
    """

    parser = argparse.ArgumentParser(description='Program to train a model over a midi dataset')
    parser.add_argument('--data', type=str, default='lmd_matched_mini', metavar='N',
                        help='The name of the data')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
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
        data_path = os.path.join('../Dataset', args.data)
        args.epochs = 2
    else:
        data_path = os.path.join('../../../../../../storage1/valentin', args.data)
    data_transformed_path = data_path + '_transformed'
    if not os.path.exists(data_transformed_path):
        os.mkdir(data_transformed_path)

    data_p = os.path.join(data_transformed_path, 'data.p')      # Pickle file with the informations of the data set
    with open(data_p, 'rb') as dump_file:
        d = pickle.load(dump_file)
        all_midi_paths = d['midi']

    ##################################
    ##################################
    ##################################

    midis_array_path = os.path.join(data_transformed_path, 'midis_array.npy')
    pl_midis_array_path = Path(midis_array_path)        # Use the library pathlib
    midis_array = np.load(midis_array_path)

    midis_array = np.reshape(midis_array, (-1, 128))

    print('midis_array : {0}'.format(midis_array.shape))

    ######################################
    ########## End Loading data ##########
    ######################################

    saved_models_path = 'saved_models'
    saved_models_pathlib = Path(saved_models_path)
    saved_models_pathlib.mkdir(parents=True, exist_ok=True)

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
    predicted_full = np.asarray(predicted_full).astype('float64')   #[:, np.newaxis, :]

    print('previous_full shape :', previous_full.shape)
    print('predicted_full shape :', predicted_full.shape)

    ###################################
    ###################################
    ###################################

    print('Definition of the graph ...')

    midi_shape = (max_len, 128)

    input_midi = tf.keras.Input(midi_shape)

    x = tf.keras.layers.LSTM(512, return_sequences=True, unit_forget_bias=True)(input_midi)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    """
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
    """
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation='softmax')(x)

    model = tf.keras.Model(input_midi, x)

    ############################

    optimizer = tf.keras.optimizers.SGD(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    #######################################
    #######################################
    #######################################

    print('Training ...')



    epoch_total = args.epochs
    batch_size = 2

    for epoch in range(1, epoch_total):
        print('Epoch:', epoch)
        model.fit(previous_full, predicted_full, batch_size=batch_size, epochs=1,
                  shuffle=True, verbose=1)

        start_index = random.randint(0, len(midis_array) - max_len - 1)

        generated_midi = midis_array[start_index: start_index + max_len]

        if ((epoch % 10) == 0):
            model.save_weights(str(saved_models_pathlib / 'my_model_weights.h5'))

            """
            # In my opinion, we don't need to generate every epoch (slower)
            for temperature in [1.2]:
                print('------ temperature:', temperature)

                for i in range(480):
                    samples = generated_midi[i:]
                    expanded_samples = np.expand_dims(samples, axis=0)
                    preds = model.predict(expanded_samples, verbose=0)[0]
                    preds = np.asarray(preds).astype('float64')

                    next_array = midi.sample(preds, temperature)

                    midi_list = []
                    midi_list.append(generated_midi)
                    midi_list.append(next_array)
                    generated_midi = np.vstack(midi_list)

                generated_midi_final = np.transpose(generated_midi, (1, 0))
                output_notes = midi.matrix_to_midi(generated_midi_final, random=0)
                midi_stream = music21.stream.Stream(output_notes)
                midi_stream.write('midi', fp='lstm_output_v1_{}_{}.mid'.format(epoch, temperature))
            """

    model.save_weights(str(saved_models_pathlib / 'my_model_weights.h5'))
    model.save(str(saved_models_pathlib / 'my_model.h5'))

    for layer in model.layers:
        lstm_weights = layer.get_weights()  # list of numpy arrays

        print('Lstm weights:', lstm_weights)

    #################################
    ####################################### Generation
    #######################################

    print('Generation...')

    start_index = random.randint(0, len(midis_array) - max_len - 1)

    for temperature in [0.7, 2.7]:
        print('------ temperature:', temperature)
        generated_midi = midis_array[start_index: start_index + max_len]
        for i in range(100):
            samples = generated_midi[i:]
            expanded_samples = np.expand_dims(samples, axis=0)
            preds = model.predict(expanded_samples, verbose=0)[0]
            preds = np.asarray(preds).astype('float64')

            next_array = midi.sample(preds, temperature)

            midi_list = []
            midi_list.append(generated_midi)
            midi_list.append(next_array)
            generated_midi = np.vstack(midi_list)

        generated_midi_final = np.transpose(generated_midi, (1, 0))
        output_notes = midi.matrix_to_midi(generated_midi_final, random=1)
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
