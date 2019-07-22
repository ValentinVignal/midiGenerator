import importlib.util
import os
import tensorflow as tf
from pathlib import Path
import pickle
import marshal
import types
import dill


class MyNN:
    """

    """

    def __init__(self):
        self.model = None
        self.loss = None
        self.model_id = None
        self.input_param = None
        self.nb_steps = None

    def new_model(self, model_id, input_param, optimizer):
        """

        :param model_id: model_name;model_param;nb_steps
        :param input_param:
        :param optimizer:
        :return: the neural network
        """

        model_name, model_param, nb_steps = model_id.split(',')
        nb_steps = int(nb_steps)

        path = os.path.join('src',
                            'NN',
                            'models',
                            'model_{0}'.format(model_name),
                            'nn_model.py')
        spec = importlib.util.spec_from_file_location('nn_model', path)
        nn_model = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(nn_model)

        # TODO: load the parameters if model_param != None

        self.model, self.loss = nn_model.create_model(
            input_param=input_param,
            model_param=model_param,
            nb_steps=nb_steps,
            optimizer=optimizer)
        self.model_id = model_id
        self.input_param = input_param
        self.nb_steps = nb_steps

    def train_seq(self, epochs, generator):
        """

        :param epochs:
        :param generator:
        :return:
        """
        self.model.fit_generator(epochs=epochs, generator=generator,
                                 shuffle=True, verbose=1)

    def save(self, path):
        """

        :param path:
        :return:
        """
        path = Path(path)
        path.mkdir(exist_ok=True, parents=True)
        self.model.save_weights(str(path / 'm_weigths.h5'))
        self.model.save(str(path / 'm.h5'))

        string_loss = dill.dumps(self.loss)

        with open(str(path / 'MyNN.p'), 'wb') as dump_file:
            pickle.dump({
                'loss': string_loss,
                'model_id': self.model_id,
                'input_param': self.input_param,
                'nb_steps': self.nb_steps
            }, dump_file)

    def load(self, path):
        """

        :param path:
        :return:
        """
        path = Path(path)
        with open(str(path / 'MyNN.p'), 'rb') as dump_file:
            d = pickle.load(dump_file)
            string_loss = d['loss']
            self.loss = dill.loads(string_loss)
            self.model_id = d['model_id']
            self.input_param = d['input_param']
            self.nb_steps = d['nb_steps']
        self.model = tf.keras.models.load_model(str(path / 'm.h5'), custom_objects={'loss_function': self.loss})

    def load_weights(self, path):
        """

        :param path:
        :return:
        """
        path = Path(path)
        self.model.load_weights(str(path / 'm_weights.h5'))

    def generate(self, input):
        """

        :param input:
        :return:
        """
        return self.model.predict(input, verbose=0)
