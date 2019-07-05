import os
from pathlib import Path
import tensorflow as tf
import src.NN.nn as nn
import pickle

from src.NN.data_generator import MySequence


class MyModel():
    """

    """

    def __init__(self, load_model=None, model_infos=None, data=None):
        # ----- General -----
        self.total_epochs = 0
        self.name = 'default_name'
        self.model = ''
        self.full_name = None
        self.get_new_full_name()

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
        self.lr = None

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

            self.new_nn_model(
                input_param=model_infos['input_param'],
                lr=getValue('lr'),
                optimizer=getValue('optimizer'),
                loss=getValue('loss')
            )
        if data is not None:
            self.load_data(data)

    def get_new_full_name(self):
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

    def load_data(self, data_transformed_path):
        """

        :return:
        """
        self.data_transformed_path = data_transformed_path
        self.data_transformed_pathlib = Path(self.data_transformed_path)
        print('data at {0} loaded'.format(data_transformed_path))

    def new_nn_model(self, input_param=None, lr=None, optimizer=None, loss=None):
        """

        :param input_param:
        :param lr:
        :param optimizer:
        :param loss:
        :return:
        """
        if input_param:
            self.input_param = input_param

        self.nn_model = nn.create_model(self.input_param)

        self.lr = lr if lr is not None else 0.01
        self.optimizer = optimizer(lr=self.lr) if optimizer is not None else tf.keras.optimizers.SGD(lr=self.lr)
        m_loss = loss if loss is not None else 'categorical_crossentropy'
        self.nn_model.compile(loss=m_loss, optimizer=self.optimizer)

    def load_model(self, id):
        """

        :param id:
        :return:
        """
        self.name, self.model, total_epochs, indice = id.split('-')
        self.total_epochs = int(total_epochs)
        self.get_new_full_name()
        path_to_load = Path('saved_models', '{0}-m({1})-e({2})-({3})'.format(self.name, self.model, self.total_epochs, indice))
        self.nn_model = tf.keras.models.load_model(str(path_to_load / 'm.h5'))
        with open(str(path_to_load / 'infos.p'), 'rb') as dump_file:
            d = pickle.load(dump_file)
            self.lr = d['nn']['lr']
            self.input_param = d['nn']['input_param']

        self.optimizer = self.nn_model.optimizer  # not sure about this part, we need to compile again ? I can load it with the pickle file
        print('Model {0} loaded'.format(id))

    def load_weights(self, id):
        self.name, self.model, total_epochs, indice = id.split('-')
        self.total_epochs = int(total_epochs)
        self.get_new_full_name()
        path_to_load = Path('saved_models', '{0}-m({1})-e({2})-({3})'.format(self.name, self.model, self.total_epochs, indice))
        self.nn_model.load_weights(str(path_to_load / 'm_weights.h5'))
        print('Weights of the {0} model loaded'.format(id))

    def train(self, epochs=50, batch=None, verbose=1, shuffle=True):
        """

        :param epochs:
        :param batch:
        :param verbose:
        :param shuffle:
        :return:
        """
        if not self.nb_files:
            with open(str(self.data_transformed_pathlib / 'infos_dataset.p'), 'rb') as dump_file:
                d = pickle.load(dump_file)
                self.nb_files = d['nb_files']

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
        print('Training...')
        self.nn_model.fit_generator(generator=self.my_sequence, epochs=epochs,
                                    shuffle=shuffle, verbose=verbose)

        # Update parameters
        self.total_epochs += epochs
        self.get_new_full_name()
        print('Training done')

    def save_model(self, path=None):
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
                }
            }, dump_file)
        print('Model saved in {0}'.format(path_to_save))

    def print_weights(self):
        for layer in self.nn_model.layers:
            lstm_weights = layer.get_weights()  # list of numpy arrays
            print('Lstm weights:', lstm_weights)

