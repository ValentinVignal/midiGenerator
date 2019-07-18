import os
from pathlib import Path
import tensorflow as tf
import src.NN.nn as nn
import pickle
import numpy as np
import progressbar

from src.NN.data_generator import MySequence
import src.global_variables as g
import src.midi.create as midi_create


class MyModel:
    """

    """

    def __init__(self, load_model=None, model_infos=None, data=None):
        """

        :param load_model: if not None, load the model
        :param model_infos: if load_model is None and model_infos is not None : create a new model with model_infos parameters
        :param data: if not None, load the data
        """
        # ----- General -----
        self.total_epochs = 0
        self.name = 'default_name'
        self.model = ''  # Id of the model used
        self.full_name = ''  # Id of this MyModel instance
        self.get_new_full_name()

        self.saved_model_path = os.path.join('saved_models', self.full_name)  # Where to saved the trained model
        self.saved_model_pathlib = Path(self.saved_model_path)

        # ----- Data -----
        self.data_transformed_path = None  # Where are the data to train on
        self.data_transformed_pathlib = None

        self.instruments = None  # List of instruments used

        # ----- MySequence -----
        self.my_sequence = None  # Instance of MySequence Generator
        self.batch = None  # Size if the batch

        # ----- Neural Network -----
        self.input_param = None  # The parameters for the neural network
        self.nn_model = None  # Our neural network
        self.optimizer = None
        self.lr = None

        self.model_id = None

        # ------ save_midi_path -----
        self.save_midis_pathlib = None  # Where to save the generated midi files

        if load_model is not None:
            self.load_model(load_model)
        elif model_infos is not None:
            def getValue(key):
                """

                :param key: key in the dictionary "model_infos"
                :return: the value in model_infos or None if it doesn't exist
                """
                value = None if key not in model_infos else model_infos[key]
                return value

            self.input_param = model_infos['input_param']
            self.new_nn_model(
                nb_steps=model_infos['nb_steps'],
                lr=getValue('lr'),
                optimizer=getValue('optimizer'),
                loss=getValue('loss')
            )
        if data is not None:
            self.load_data(data)

    def get_full_name(self, i):
        """

        :param i: index
        :return: set up the full name and the path to save the trained model
        """
        full_name = '{0}-m({1})-e({2})-({3})'.format(self.name, self.model, self.total_epochs, i)
        saved_model_path = os.path.join('saved_models', full_name)
        saved_model_pathlib = Path(saved_model_path)
        self.full_name = full_name
        self.saved_model_path = saved_model_path
        self.saved_model_pathlib = saved_model_pathlib
        print('Get full_name : {0}'.format(self.full_name))

    def get_new_full_name(self):
        """

        :return: set up a new unique full name and the corresponding path to save the trained model
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
        self.saved_model_path = saved_model_path
        self.saved_model_pathlib = saved_model_pathlib
        self.full_name = full_name
        print('Got new full_name : {0}'.format(self.full_name))

        self.save_midis_pathlib = None

    def load_data(self, data_transformed_path):
        """

        :return: load the data
        """
        self.data_transformed_path = data_transformed_path
        self.data_transformed_pathlib = Path(self.data_transformed_path)
        self.input_param = {}
        with open(str(self.data_transformed_pathlib / 'infos_dataset.p'), 'rb') as dump_file:
            d = pickle.load(dump_file)
            self.input_param['input_size'] = d['input_size']
            self.input_param['nb_instruments'] = d['nb_instruments']
            self.instruments = d['instruments']
        print('data at {0} loaded'.format(data_transformed_path))

    def new_nn_model(self, model_id, nb_steps, lr=None, optimizer=None, loss=None):
        """

        :param model_id:
        :param nb_steps: Size for the RNN
        :param lr:
        :param optimizer:
        :param loss:
        :return: set up the neural network
        """
        try:
            _ = self.input_param['input_size']
            _ = self.input_param['nb_instruments']
        except KeyError:
            print('Load the data before creating a new model')

        self.model_id = model_id
        self.input_param['nb_steps'] = nb_steps

        self.nn_model = nn.create_nn_model(
            model_id=self.model_id,
            input_param=self.input_param)

        self.lr = lr if lr is not None else 0.001
        self.optimizer = optimizer(
            lr=self.lr) if optimizer is not None \
            else tf.keras.optimizers.Adam(lr=self.lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.1,
                                          amsgrad=False)  # tf.keras.optimizers.SGD(lr=self.lr)
        m_loss = loss if loss is not None else 'categorical_crossentropy'
        self.nn_model.compile(loss=m_loss, optimizer=self.optimizer)

    def load_model(self, id, keep_name=True):
        """

        :param id: id of the model to load
        :param keep_name: if true keep the name, if not, get a new index at the and of the full name
        :return: load a model
        """
        self.name, self.model, total_epochs, indice = id.split('-')
        self.total_epochs = int(total_epochs)
        if keep_name:
            self.get_full_name(indice)
        else:
            self.get_new_full_name()
        path_to_load = Path('saved_models',
                            '{0}-m({1})-e({2})-({3})'.format(self.name, self.model, self.total_epochs, indice))
        self.nn_model = tf.keras.models.load_model(str(path_to_load / 'm.h5'))
        with open(str(path_to_load / 'infos.p'), 'rb') as dump_file:
            d = pickle.load(dump_file)
            self.lr = d['nn']['lr']
            self.input_param = d['nn']['input_param']
            self.instruments = d['instruments']
        self.optimizer = self.nn_model.optimizer  # not sure about this part, we need to compile again ? I can load it with the pickle file
        print('Model {0} loaded'.format(id))

    def load_weights(self, id, keep_name=True):
        """

        :param id: id of the model to load
        :param keep_name: if true keep the name, if not, get a new index at the and of the full name
        :return: load the weights of a model
        """
        self.name, self.model, total_epochs, indice = id.split('-')
        self.total_epochs = int(total_epochs)
        if keep_name:
            self.get_full_name(indice)
        else:
            self.get_new_full_name()
        path_to_load = Path('saved_models',
                            '{0}-m({1})-e({2})-({3})'.format(self.name, self.model, self.total_epochs, indice))
        self.nn_model.load_weights(str(path_to_load / 'm_weights.h5'))
        print('Weights of the {0} model loaded'.format(id))

    def train(self, epochs=50, batch=None, verbose=1, shuffle=True):
        """

        :param epochs:
        :param batch:
        :param verbose:
        :param shuffle:
        :return: train the model
        """

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
                path=str(self.data_transformed_pathlib),
                nb_steps=self.input_param['nb_steps'],
                batch_size=self.batch
            )

        # Actual train
        print('Training...')
        self.nn_model.fit_generator(generator=self.my_sequence, epochs=epochs,
                                    shuffle=shuffle, verbose=verbose)

        # Update parameters
        self.total_epochs += epochs
        self.get_new_full_name()
        print('Training done')

    def save_model(self, path=None):
        """

        :param path: path were to save the model, if Nonem it will be at self.saved_model_path
        :return:
        """
        path_to_save = self.saved_model_pathlib if path is None else Path(path)
        path_to_save.mkdir(parents=True, exist_ok=True)  # Creation of this folder
        self.saved_model_pathlib.mkdir(parents=True, exist_ok=True)
        self.nn_model.save_weights(str(path_to_save / 'm_weights.h5'))
        self.nn_model.save(str(path_to_save / 'm.h5'))
        with open(str(path_to_save / 'infos.p'), 'wb') as dump_file:
            pickle.dump({
                'name': self.name,
                'model': self.model,
                'full_name': self.full_name,
                'nn': {
                    'epochs': self.total_epochs,
                    'input_param': self.input_param,
                    'lr': self.lr
                    # 'optimizer': self.optimizer,
                },
                'instruments': self.instruments
            }, dump_file)
        print('Model saved in {0}'.format(path_to_save))

    def print_weights(self):
        """
        Print the weights
        :return:
        """
        for layer in self.nn_model.layers:
            lstm_weights = layer.get_weights()  # list of numpy arrays
            print('Lstm weights:', lstm_weights)

    def get_new_save_midis_path(self, path=None):
        """
        set up a new save midi path
        :param path:
        :return:
        """
        if path is None:
            i = 0
            m_str = '{0}-generation({1})'.format(self.full_name, i)
            while Path('generated_midis', m_str).exists():
                i += 1
                m_str = '{0}-generation({1})'.format(self.full_name, i)
            self.save_midis_pathlib = Path('generated_midis', m_str)
        else:
            self.save_midis_pathlib = Path(path)
        print('new save path for midi files :', str(self.save_midis_pathlib))

    def generate(self, seed=None, length=None, new_save_path=None):
        """
        Generate midi file from the seed and the trained model
        :param seed: seed for the generation
        :param length: Length of th generation
        :param new_save_path:
        :return:
        """
        # --- Verify the inputs ---
        if seed is None:
            seed = np.random.uniform(
                low=g.min_value,
                high=g.max_value,
                size=(self.input_param['nb_instruments'], self.input_param['nb_steps'], self.input_param['input_size']))
        length = length if length is not None else 200
        # For save midi path
        if type(new_save_path) is str or (
                type(new_save_path) is bool and new_save_path) or (
                new_save_path is None and self.save_midis_pathlib is None):
            self.get_new_save_midis_path(path=new_save_path)
        # --- Done Verifying the inputs ---

        self.save_midis_pathlib.mkdir(parents=True, exist_ok=True)
        print('Start generating ...')
        generated = seed
        bar = progressbar.ProgressBar(maxval=length,
                                      widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage(), ' ',
                                               progressbar.ETA()])
        bar.start()  # To see it working
        for l in range(length):
            samples = generated[:, np.newaxis, l:, :]
            # expanded_samples = np.expand_dims(samples, axis=0)
            preds = self.nn_model.predict(list(samples), verbose=0)
            preds = np.asarray(preds).astype('float64')  # (nb_instruments, 1, 128)

            # next_array = np.zeros(preds.shape)
            # for inst in range(next_array.shape[0]):
            #     next_array[inst] = midi_create.sample(preds[inst][0], temperature)[np.newaxis, :]
            next_array = preds  # Without temperature

            # generated_list = []
            # generated_list.append(generated)
            # generated_list.append(next_array)
            # generated = np.vstack(generated_list)
            generated = np.concatenate((generated, next_array), axis=1)  # (nb_instruments, nb_steps, 128)

            bar.update(l + 1)
        bar.finish()

        generated_midi_final = np.transpose(generated, (0, 2, 1))  # (nb_instruments, 128, nb_steps)
        output_notes_list = midi_create.matrix_to_midi(generated_midi_final, instruments=self.instruments)
        # --- find the name for the midi_file ---
        i = 0
        m_str = "lstm_out_t({0}).mid".format(i)
        while (self.save_midis_pathlib / m_str).exists():
            i += 1
            m_str = "lstm_out_({0}).mid".format(i)
        path_to_save = str(self.save_midis_pathlib / m_str)

        # Saving the midi file
        midi_create.save_midi(output_notes_list=output_notes_list, instruments=self.instruments, path=path_to_save)
        print('Done Generating')
