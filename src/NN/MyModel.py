import os
from pathlib import Path
import pickle
import numpy as np
import progressbar
import random

from src.NN.nn import MyNN
from src.NN.data_generator import MySequence
import src.midi.create as midi_create
import src.image.pianoroll as pianoroll
import src.text.summary as summary


class MyModel:
    """

    """

    def __init__(self, name='default_name', load_model=None, model_infos=None, data=None):
        """

        :param name: The name of the model
        :param load_model: if not None, load the model
        :param model_infos: if load_model is None and model_infos is not None : create a new model with model_infos parameters
        :param data: if not None, load the data
        """
        # ----- General -----
        self.total_epochs = 0
        self.name = name
        self.model_id = ''  # Id of the model used
        self.full_name = ''  # Id of this MyModel instance
        self.get_new_full_name()

        self.saved_model_path = os.path.join('saved_models', self.full_name)  # Where to saved the trained model
        self.saved_model_pathlib = Path(self.saved_model_path)

        # ----- Data -----
        self.data_transformed_path = None  # Where are the data to train on
        self.data_transformed_pathlib = None

        self.data_seed_pathlib = None

        self.instruments = None  # List of instruments used
        self.notes_range = None

        # ----- MySequence -----
        self.my_sequence = None  # Instance of MySequence Generator
        self.batch = None  # Size if the batch

        # ----- Neural Network -----
        self.input_param = None  # The parameters for the neural network
        self.my_nn = None  # Our neural network

        # ------ save_midi_path -----
        self.save_midis_pathlib = None  # Where to save the generated midi files

        if data is not None:
            self.load_data(data)

    @classmethod
    def from_model(cls, id, name='defaultName', data=None):
        myModel = cls(name=name, data=data)
        myModel.load_model(id=id)
        return myModel

    @classmethod
    def with_model(cls, model_infos, name='defaultName', data=None):
        myModel = cls(name=name, data=data)

        def get_value(key):
            """

            :param key: key in the dictionary "model_infos"
            :return: the value in model_infos or None if it doesn't exist
            """
            value = None if key not in model_infos else model_infos[key]
            return value

        myModel.input_param = model_infos['input_param']
        myModel.model_id = model_infos['model_id']
        myModel.new_nn_model(
            model_id=model_infos['model_id'],
            opt_param=get_value('opt_param'),
        )
        return myModel

    def get_full_name(self, i):
        """

        :param i: index
        :return: set up the full name and the path to save the trained model
        """
        full_name = '{0}-m({1})-e({2})-({3})'.format(self.name, self.model_id, self.total_epochs, i)
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
        full_name = '{0}-m({1})-e({2})-({3})'.format(self.name, self.model_id, self.total_epochs, i)
        saved_model_path = os.path.join('saved_models', full_name)
        saved_model_pathlib = Path(saved_model_path)
        while saved_model_pathlib.exists():
            i += 1
            full_name = '{0}-m({1})-e({2})-({3})'.format(self.name, self.model_id, self.total_epochs, i)
            saved_model_path = os.path.join('saved_models', full_name)
            saved_model_pathlib = Path(saved_model_path)
        self.saved_model_path = saved_model_path
        self.saved_model_pathlib = saved_model_pathlib
        self.full_name = full_name
        print('Got new full_name : {0}'.format(self.full_name))

        self.save_midis_pathlib = None

    def set_name(self, name=None):
        """

        :param name:
        :return:
        """
        self.name = self.name if name is None else name
        self.get_new_full_name()

    def load_data(self, data_transformed_path=None):
        """

        :return: load the data
        """
        self.data_transformed_path = data_transformed_path if data_transformed_path is not None else self.data_transformed_path
        self.data_transformed_pathlib = Path(self.data_transformed_path)
        self.input_param = {}
        with open(str(self.data_transformed_pathlib / 'infos_dataset.p'), 'rb') as dump_file:
            d = pickle.load(dump_file)
            self.input_param['input_size'] = d['input_size']
            self.input_param['nb_instruments'] = d['nb_instruments']
            self.instruments = d['instruments']
            self.notes_range = d['notes_range']
        print('data at {0} loaded'.format(data_transformed_path))

    def new_nn_model(self, model_id, opt_param=None):
        """

        :param model_id: modelName;modelParam;nbSteps
        :param opt_param:
        :return: set up the neural network
        """
        try:
            _ = self.input_param['input_size']
            _ = self.input_param['nb_instruments']
        except KeyError:
            print('Load the data before creating a new model')

        self.model_id = model_id
        self.get_new_full_name()

        opt_param = {'lr': 0.01, 'name': 'adam'} if opt_param is None else opt_param

        self.my_nn = MyNN()
        self.my_nn.new_model(model_id=self.model_id,
                             input_param=self.input_param,
                             opt_param=opt_param)

    def load_model(self, id, keep_name=True):
        """

        :param id: id of the model to load
        :param keep_name: if true keep the name, if not, get a new index at the and of the full name
        :return: load a model
        """
        self.name, self.model_id, total_epochs, indice = id.split('-')
        self.total_epochs = int(total_epochs)
        if keep_name:
            self.get_full_name(indice)
        else:
            self.get_new_full_name()
        path_to_load = Path('saved_models',
                            '{0}-m({1})-e({2})-({3})'.format(self.name, self.model_id, self.total_epochs, indice))
        self.my_nn = MyNN()
        self.my_nn.load(str(path_to_load / 'MyNN'))
        with open(str(path_to_load / 'infos.p'), 'rb') as dump_file:
            d = pickle.load(dump_file)
            self.input_param = d['nn']['input_param']
            self.instruments = d['instruments']
            self.data_seed_pathlib = Path(d['data_seed_pathlib'])
        print('Model {0} loaded'.format(id))

    def load_weights(self, id, keep_name=True):
        """

        :param id: id of the model to load
        :param keep_name: if true keep the name, if not, get a new index at the and of the full name
        :return: load the weights of a model
        """
        self.name, self.model_id, total_epochs, indice = id.split('-')
        self.total_epochs = int(total_epochs)
        if keep_name:
            self.get_full_name(indice)
        else:
            self.get_new_full_name()
        path_to_load = Path('saved_models',
                            '{0}-m({1})-e({2})-({3})'.format(self.name, self.model_id, self.total_epochs, indice))
        self.my_nn.load_weights(str(path_to_load / 'MyNN.h5'))
        print('Weights of the {0} model loaded'.format(id))

    def train(self, epochs=None, batch=None):
        """

        :param epochs:
        :param batch:
        :param verbose:
        :param shuffle:
        :return: train the model
        """

        # Do we have to create a new MySequence Object ?
        flag_new_sequence = False
        epochs = 50 if epochs is None else epochs
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
                nb_steps=int(self.model_id.split(',')[2]),
                batch_size=self.batch
            )

        # Actual train
        print('Training...')
        self.my_nn.train_seq(epochs=epochs, generator=self.my_sequence)

        # Update parameters
        self.total_epochs += epochs
        self.data_seed_pathlib = self.data_transformed_pathlib
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
        self.my_nn.save(str(path_to_save / 'MyNN'))
        with open(str(path_to_save / 'infos.p'), 'wb') as dump_file:
            pickle.dump({
                'name': self.name,
                'model_id': self.model_id,
                'full_name': self.full_name,
                'nn': {
                    'epochs': self.total_epochs,
                    'input_param': self.input_param,
                },
                'instruments': self.instruments,
                'data_seed_pathlib': str(self.data_seed_pathlib),
                'notes_range': self.notes_range
            }, dump_file)
        summary.summarize_train(path_to_save, **{
            'full_name': self.full_name,
            'epochs': self.total_epochs,
            'input_param': self.input_param,
            'instruments': self.instruments,
            'notes_range': self.notes_range
        })

        print('Model saved in {0}'.format(path_to_save))

    def print_weights(self):
        """
        Print the weights
        :return:
        """
        for layer in self.my_nn.model.layers:
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

    def generate(self, seed=None, length=None, new_save_path=None, save_images=False):
        """
        Generate midi file from the seed and the trained model
        :param seed: seed for the generation
        :param length: Length of th generation
        :param new_save_path:
        :param save_images:
        :return:
        """
        # --- Verify the inputs ---
        nb_steps = int(self.model_id.split(',')[2])
        if type(seed) is list:
            pass
        elif seed is None:
            seed = 1
        elif type(seed) is int:
            seed = self.get_seed(nb_steps=nb_steps, number=seed)
        length = length if length is not None else 200
        # For save midi path
        if type(new_save_path) is str or (
                type(new_save_path) is bool and new_save_path) or (
                new_save_path is None and self.save_midis_pathlib is None):
            self.get_new_save_midis_path(path=new_save_path)
        # --- Done Verifying the inputs ---

        self.save_midis_pathlib.mkdir(parents=True, exist_ok=True)
        print('Start generating ...')
        for s in range(len(seed)):
            print('Generation {0}/{1}'.format(s + 1, len(seed)))
            generated = seed[s]
            bar = progressbar.ProgressBar(maxval=length,
                                          widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage(), ' ',
                                                   progressbar.ETA()])
            bar.start()  # To see it working
            for l in range(length):
                samples = generated[:, np.newaxis, l:, :]  # (nb_instruments, 1, nb_steps, 88, 2)
                # expanded_samples = np.expand_dims(samples, axis=0)
                preds = self.my_nn.generate(input=list(samples))
                preds = np.asarray(preds).astype('float64')  # (nb_instruments, 1, 88, 2)
                next_array = preds  # Without temperature
                generated = np.concatenate((generated, next_array), axis=1)  # (nb_instruments, nb_steps, 88, 2)

                bar.update(l + 1)
            bar.finish()

            generated_midi_final = np.transpose(generated, (0, 2, 1, 3))  # (nb_instruments, 88, nb_steps, 2)
            output_notes_list = midi_create.matrix_to_midi(generated_midi_final, instruments=self.instruments,
                                                           notes_range=self.notes_range)

            # --- find the name for the midi_file ---
            i = 0
            m_str = "lstm_out_({0}).mid".format(i)
            while (self.save_midis_pathlib / m_str).exists():
                i += 1
                m_str = "lstm_out_({0}).mid".format(i)
            path_to_save = str(self.save_midis_pathlib / m_str)
            path_to_save_img = str(self.save_midis_pathlib / 'lstm_out_({0}).jpg'.format(i))

            # Saving the midi file
            midi_create.save_midi(output_notes_list=output_notes_list, instruments=self.instruments, path=path_to_save)
            if save_images:
                pianoroll.save_pianoroll(array=generated_midi_final,
                                         path=path_to_save_img,
                                         seed_length=nb_steps,
                                         instruments=self.instruments)

        print('Done Generating')

    def get_seed(self, nb_steps, number=1):
        """

        :param nb_steps:
        :param number:
        :return:
        """
        seeds = []
        with open(self.data_seed_pathlib / 'infos_dataset.p', 'rb') as dump_file:
            d = pickle.load(dump_file)
            all_shapes = d['all_shapes']
        for i in range(number):
            array_list = np.load(str(self.data_seed_pathlib / 'npy' / '{0}.npy'.format(
                random.randint(0, len(all_shapes) - 1)
            )), allow_pickle=True).item()['list']
            array = array_list[random.randint(0, len(array_list) - 1)]
            start = random.randint(0, len(array) - nb_steps - 1)
            seed = array[start: start + nb_steps]  # (nb_steps, nb_intruments, input_size, 2)
            seed = np.transpose(seed, (1, 0, 2, 3))  # (nb_instruments, nb_steps, input_size, 2)
            seeds.append(seed)
        return seeds
