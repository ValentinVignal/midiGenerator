import importlib.util
import os
import tensorflow as tf
from pathlib import Path
import pickle
import dill
import json


class MyNN:
    """

    """

    def __init__(self):
        self.model = None
        self.loss = None
        self.model_id = None
        self.input_param = None
        self.nb_steps = None

        # Spare GPU
        MyNN.allow_growth()

    def new_model(self, model_id, input_param, opt_param):
        """

        :param model_id: model_name;model_param;nb_steps
        :param input_param:
        :param opt_param: {'lr', 'name'}
        :return: the neural network
        """

        model_name, model_param_s, nb_steps = model_id.split(',')
        nb_steps = int(nb_steps)

        # Load .py file with the model in it
        path = os.path.join('src',
                            'NN',
                            'models',
                            'model_{0}'.format(model_name),
                            'nn_model.py')
        spec = importlib.util.spec_from_file_location('nn_model', path)
        nn_model = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(nn_model)

        # Load model param .json file
        json_path = os.path.join('src',
                                 'NN',
                                 'models',
                                 'model_{0}'.format(model_name),
                                 '{0}.json'.format(model_param_s))
        with open(json_path) as json_file:
            model_param = json.load(json_file)

        optimizer = MyNN.create_optimizer(opt_param)

        self.model, self.loss = nn_model.create_model(
            input_param=input_param,
            model_param=model_param,
            nb_steps=nb_steps,
            optimizer=optimizer)
        self.model_id = model_id
        self.input_param = input_param
        self.nb_steps = nb_steps

    @staticmethod
    def create_optimizer(opt_param):
        """

        :param opt_param:
        :return:
        """
        name = opt_param['name']
        lr = opt_param['lr']

        if name == 'adam':
            return tf.keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.4, amsgrad=False)
        elif name == 'sgd':
            return tf.keras.optimizers.SGD(lr=lr)

    def train_seq(self, epochs, generator):
        """

        :param epochs:
        :param generator:
        :return:
        """
        self.allow_growth()
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

    @staticmethod
    def allow_growth():
        """

        :return:
        """
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        tf.keras.backend.set_session(sess)

    @staticmethod
    def choose_gpu(gpu):
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu

