import argparse
import os
import music21
import numpy as np
from pathlib import Path
import tensorflow as tf
import random
import bottleneck
import pickle

import src.midi as midi
import src.NN.nn as nn
from src.NN.data_generator import MySequence


class Trainer():
    """

    """

    def __init__(self):
        # ----- General -----
        self.total_epochs = 0
        self.name = 'default_name'
        self.model = ''
        self.full_name = self.return_full_name()

        self.saved_model_path = os.path.join('saved_models', self.full_name)
        self.saved_model_pathlib = Path(self.saved_model_path)

        # ----- Data -----
        self.data_transformed_path = None
        self.data_transformed_pathlib = None

        self.nb_files = None

        # ----- MySequence -----
        self.my_sequence = None
        self.batch = None

        # ----- Neural Network -----
        self.input_param = None
        self.nn_model = None
        self.optimizer = None

    def return_full_name(self):
        """

        :return: set self.full name
        """
        i = 0
        full_name = '{0}-m({1})-e({2})-({3})'.format(self.name, self.model, self.total_epochs, i)
        saved_model_path = os.path.join('saved_models', full_name)
        saved_model_pathlib = Path(saved_model_path)
        while saved_model_pathlib.exists():
            i += 1
            full_name = '{0}-m({1})-e({2})-({3})'.format(self.name, self.model, self.total_epochs, i)
            saved_model_path = os.path.join('saved_models', full_name)
            saved_model_pathlib = Path(saved_model_path)
        return full_name

    def save_infos(self, path=None):
        """

        :return:
        """
        path_to_save = self.saved_model_pathlib if path is None else Path(path)
        path_to_save.mkdir(parents=True, exist_ok=True)  # Creation of this folder
        with open(str(path_to_save / 'infos.p'), 'wb') as dump_file:
            pickle.dump({
                'name': self.name,
                'model': self.model,
                'epochs': self.total_epochs,
                'full_name': self.full_name
            }, dump_file)

    def load_data(self, data_transformed_path):
        """

        :return:
        """
        self.data_transformed_path = data_transformed_path
        self.data_transformed_pathlib = Path(self.data_transformed_path)

    def new_nn_model(self, input_param=None, lr=0.01, optimizer=None, loss='categorical_crossentropy'):
        """

        :param input_param: parameters of the input f the neural network
        :return: Create the neural network
        """
        if input_param:
            self.input_param = input_param
        self.nn_model = nn.my_model(self.input_param)

        self.optimizer = optimizer(lr=lr) if optimizer else tf.keras.optimizers.SGD(lr=lr)
        self.nn_model.compile(loss=loss, optimizer=self.optimizer)

    def train(self, epochs=50, batch=None, verbose=1, shuffle=True):
        """

        :param epochs:
        :param batch:
        :param verbose:
        :param suffle:
        :return:
        """
        if not self.nb_files:
            with (str(self.data_transformed_pathlib / 'data.p'), 'rb') as dump_file:
                d = pickle.load(dump_file)
                self.nb_files = len(d['midi'])

        # Do we have to create a new MySequence Object ?
        flag_new_sequence = False
        if batch is None and self.batch is None:
            self.batch = 1,
            flag_new_sequence = True
        if batch is not None and batch != self.batch:
            self.batch = batch
            flag_new_sequence = True
        if self.my_sequence is None:
            flag_new_sequence = True

        if flag_new_sequence:
            self.my_sequence = MySequence(
                nb_files=self.nb_files,
                npy_path=str(self.data_transformed_pathlib / 'npy'),
                nb_step=self.input_param['nb_steps'],
                batch_size=self.batch
            )

        # Actual train
        self.nn_model.fit_generator(generator=self.my_sequence, epochs=epochs,
                                    shuffle=shuffle, verbose=verbose)

        # Update parameters
        self.total_epochs += epochs

    def save_weights(self, path=None):
        path_to_save = self.saved_model_pathlib if path is None else Path(path)
        path_to_save.mkdir(parents=True, exist_ok=True)  # Creation of this folder
        self.saved_model_pathlib.mkdir(parents=True, exist_ok=True)
        self.nn_model.save_weights(str(path_to_save / 'm_weights.h5'))
        self.nn_model.save(str(path_to_save / 'm.h5'))

    def print_weights(self):
        for layer in self.nn_model.layers:
            lstm_weights = layer.get_weights()  # list of numpy arrays

            print('Lstm weights:', lstm_weights)


def main():
    """
        Entry point
    """

    parser = argparse.ArgumentParser(description='Program to train a model over a midi dataset')
    parser.add_argument('-d', '--data', type=str, default='lmd_matched_mini', metavar='N',
                        help='The name of the data')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 5)')
    parser.add_argument('-b', '--batch', type=int, default=1,
                        help='The number of the batchs')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--pc', action='store_true', default=False,
                        help='to work on a small computer with a cpu')
    parser.add_argument('--seed', type=int, default=1234, metavar='S',
                        help='random seed (default: 1234)')
    parser.add_argument('--log-interval', type=int, default=5, metavar='N',
                        help='how many batch to wait before logging training status')
    parser.add_argument('-n', '--name', type=str, default='default_name',
                        help='how many batch to wait before logging training status')
    load_group = parser.add_mutually_exclusive_group()
    load_group.add_argument('-m', '--model', type=str, default='1',
                            help='The model of the Neural Network used for the interpolation')
    load_group.add_argument('-l', '--load', type=str, default='',
                            help='The name of the trained model to load')
    parser.add_argument('--gpu', type=str, default='0',
                        help='What GPU to use')

    args = parser.parse_args()

    if args.pc:
        data_path = os.path.join('../Dataset', args.data)
        args.epochs = 2
    else:
        data_path = os.path.join('../../../../../../storage1/valentin', args.data)
    data_transformed_path = data_path + '_transformed'

    trainer = Trainer()
    trainer.load_data(data_transformed_path=data_transformed_path)

    input_param = {
        'nb_steps': 18,
        'input_size': 128
    }

    trainer.new_nn_model(input_param=input_param)
    trainer.train(epochs=args.epochs, batch=args.batch, verbose=1, shuffle=True)
    trainer.save_weights()
    trainer.save_infos()
    trainer.print_weights()

    """
    total_epochs = args.epochs
    name = args.name
    model = args.model

    def save_infos():
        i = 0
        full_name = '{0}-m({1})-e({2})-({3})'.format(name, model, total_epochs, i)
        saved_model_path = os.path.join('saved_models', full_name)
        saved_model_pathlib = Path(saved_model_path)
        while saved_model_pathlib.exists():
            i += 1
            full_name = '{0}-m({1})-e({2})-({3})'.format(name, model, total_epochs, i)
            saved_model_path = os.path.join('saved_models', full_name)
            saved_model_pathlib = Path(saved_model_path)
        saved_model_pathlib.mkdir(parents=True, exist_ok=True)  # Creation of this folder
        with open(str(saved_model_pathlib / 'info.p'), 'wb') as dump_file:
            pickle.dump({
                'name': name,
                'model': model,
                'epochs': total_epochs,
                'full_name': full_name
            }, dump_file)

    save_infos()

    ##################################
    ##################################
    ##################################

    # folder where are the .npy files
    npy_path = os.path.join(data_transformed_path, 'npy')
    npy_pathlib = Path(npy_path)

    max_len = 18  # how many column will take account to predict next column.
    step = 1  # step size.

    input_param = {
        'nb_steps': 18,
        'input_size': 128
    }
    model = nn.my_model(input_param=input_param)

    my_sequence = MySequence(nb_files=0, npy_path=npy_path, nb_step=max_len, batch_size=args.batch)

    ############################

    optimizer = tf.keras.optimizers.SGD(lr=args.lr)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    #######################################
    #######################################
    #######################################

    print('Training ...')

    model.fit_generator(generator=my_sequence, epochs=args.epochs,
                        shuffle=True, verbose=1)
    """

    """
    for epoch in range(1, epoch_total):
        print('Epoch:', epoch)
        model.fit(previous_full, predicted_full, batch_size=batch_size, epochs=1,
                  shuffle=True, verbose=1)

        start_index = random.randint(0, len(midis_array) - max_len - 1)

        generated_midi = midis_array[start_index: start_index + max_len]

        if ((epoch % 10) == 0):
            model.save_weights(str(saved_models_pathlib / 'my_model_weights.h5'))

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

    """
    model.save_weights(str(saved_model_pathlib / 'm_weights.h5'))
    model.save(str(saved_model_pathlib / 'm.h5'))

    for layer in model.layers:
        lstm_weights = layer.get_weights()  # list of numpy arrays

        print('Lstm weights:', lstm_weights)
    """

    #################################
    ####################################### Generation
    #######################################

    """
    print('Generation...')

    generated_midis_path = 'generated_midis'
    generated_midis_pathlib = Path(generated_midis_path)
    generated_midis_pathlib.mkdir(parents=True, exist_ok=True)

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
        midi_file_name = (str(generated_midis_pathlib / 'lstm_out_{}.mid'.format(temperature)))
        midi_stream.write('midi', fp=midi_file_name)
        parsed = music21.converter.parse(midi_file_name)
        for part in parsed.parts:
            part.insert(0, music21.instrument.Piano())
        parsed.write('midi', fp=midi_file_name)

    # To see values

    z = -bottleneck.partition(-preds, 20)[:20]
    print(z)
    print('max:', np.max(preds))

    model.save_weights(str(saved_models_pathlib / 'my_model_weights.h5'))
    model.save(str(saved_models_pathlib / 'my_model.h5'))

    print('Done')
    """


if __name__ == '__main__':
    # create a separate main function because original main function is too mainstream
    main()
