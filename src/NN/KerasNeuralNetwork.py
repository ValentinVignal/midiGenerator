import os
import tensorflow as tf
from tensorflow.python.keras.callbacks import TensorBoard
from pathlib import Path
import pickle
import dill
import json
import math
from time import time
import progressbar
import numpy as np

import src.NN.losses as nn_losses
import src.global_variables as g
import src.NN.sequences.TrainValSequence as tv_sequences

from . import models

K = tf.keras.backend


class KerasNeuralNetwork:
    """

    """

    def __init__(self):
        self.model_id = None
        self.input_param = None
        self.nb_steps = None
        self.step_length = None
        # --- TF ---
        self.model = None
        self.losses = None
        self.loss_lambdas = None
        self.opt_param = None
        self.decay = None
        self.type_loss = None

        self.model_options = None

        # Spare GPU
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        self.allow_growth()

        self.tensorboard = TensorBoard(log_dir='tensorboard\\{0}'.format(time()))

    def new_model(self, model_id, input_param, opt_param, type_loss=None, step_length=1, model_options={}):
        """

        :param model_id: model_name;model_param;nb_steps
        :param input_param:
        :param opt_param: {'lr', 'name'}
        :param type_loss:
        :param step_length:
        :param model_options:
        :return: the neural network
        """

        if type_loss is not None:
            self.type_loss = type_loss
        elif self.type_loss is None:
            self.type_loss = g.type_loss
        self.step_length = step_length

        model_name, model_param_s, nb_steps = model_id.split(g.split_model_id)
        nb_steps = int(nb_steps)

        nn_model = models.from_name[model_name]

        # Load model param .json file
        json_path = os.path.join('src',
                                 'NN',
                                 'models',
                                 model_name,
                                 '{0}.json'.format(model_param_s))
        with open(json_path) as json_file:
            model_param = json.load(json_file)

        self.opt_param = opt_param
        optimizer, self.decay = KerasNeuralNetwork.create_optimizer(**self.opt_param)

        self.model, self.losses, self.loss_lambdas = nn_model.create_model(
            input_param=input_param,
            model_param=model_param,
            nb_steps=nb_steps,
            optimizer=optimizer,
            step_length=step_length,
            type_loss=self.type_loss,
            model_options=model_options)
        self.model_id = model_id
        self.input_param = input_param
        self.nb_steps = nb_steps
        self.model_options = model_options

    @staticmethod
    def create_optimizer(**kwargs):
        """

        :param kwargs:
        :return:
        """
        # ----- Default values -----
        opt_param = {
            'name': 'adam',
            'lr': 0.01,
            'drop': 0.5,
            'epochs_drop': 10
        }
        opt_param.update(kwargs)

        # ----- Optimizer -----
        optimizer = None
        if opt_param['name'] == 'adam':
            optimizer = tf.keras.optimizers.Adam(lr=opt_param['lr'], beta_1=0.9, beta_2=0.999, epsilon=None,
                                                 amsgrad=False)
        elif opt_param['name'] == 'sgd':
            optimizer = tf.keras.optimizers.SGD(lr=opt_param['lr'])

        # ----- Decay -----
        step_decay = dill.loads(
            KerasNeuralNetwork.decay_func(lr_init=opt_param['lr'], drop=opt_param['drop'],
                                          epochs_drop=opt_param['epochs_drop']))
        # lrate = tf.keras.callbacks.LearningRateScheduler(step_decay)
        # callback_list = [lrate]

        return optimizer, step_decay

    @staticmethod
    def decay_func(lr_init, **kwargs):
        def step_decay(epoch):
            lrate = lr_init * math.pow(kwargs['drop'], math.floor(epoch / kwargs['epochs_drop']))
            return lrate

        return dill.dumps(step_decay)

    def train_seq(self, epochs, generator, callbacks=[], verbose=1, validation=0.0):
        """

        :param epochs:
        :param generator:
        :param callbacks:
        :param verbose:
        :param validation:
        :return:
        """
        callback_list = [tf.keras.callbacks.LearningRateScheduler(self.decay), self.tensorboard] + callbacks

        if validation > 0:
            generator_train, generator_valid = tv_sequences.get_train_valid_sequence(generator,
                                                                                     validation_split=validation)

            a = self.model.fit_generator(epochs=epochs, generator=generator_train, validation_data=generator_valid,
                                         shuffle=True, verbose=verbose, callbacks=callback_list)
        else:  # So it won't print a lot of lines for nothing
            a = self.model.fit_generator(epochs=epochs, generator=generator,
                                         shuffle=True, verbose=verbose, callbacks=callback_list)

        return a.history

    def evaluate(self, generator, verbose=1):
        evaluation = self.model.evaluate_generator(generator=generator, verbose=verbose)
        return evaluation

    def save(self, path):
        """

        :param path:
        :return:
        """
        path = Path(path)
        path.mkdir(exist_ok=True, parents=True)
        string_loss = dill.dumps(self.losses)
        string_decay = dill.dumps(self.decay)
        with open(str(path / 'weights.p'), 'wb') as dump_file:
            pickle.dump({
                'weights': self.model.get_weights()
            }, dump_file)

        with open(str(path / 'MyNN.p'), 'wb') as dump_file:
            pickle.dump({
                'loss': string_loss,
                'loss_lambdas': self.loss_lambdas,
                'model_id': self.model_id,
                'input_param': self.input_param,
                'nb_steps': self.nb_steps,
                'decay': string_decay,
                'opt_param': self.opt_param,
                'type_loss': self.type_loss,
                'step_length': self.step_length,
                'model_options': self.model_options,
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
            self.losses = dill.loads(string_loss)
            self.loss_lambdas = d['loss_lambdas']
            self.model_id = d['model_id']
            self.input_param = d['input_param']
            self.nb_steps = d['nb_steps']
            self.decay = dill.loads(d['decay'])
            self.opt_param = d['opt_param']
            self.type_loss = d['type_loss']
            self.step_length = d['step_length']

        optimizer, self.decay = KerasNeuralNetwork.create_optimizer(**self.opt_param)
        metrics = [nn_losses.acc_act, nn_losses.mae_dur]
        self.model = tf.keras.models.load_model(str(path / 'm.h5'),
                                                custom_objects={'losses': self.losses,
                                                                'loss_function': nn_losses.choose_loss(self.type_loss)(
                                                                    *self.loss_lambdas),
                                                                'optimizer': optimizer,
                                                                'acc_act': nn_losses.acc_act,
                                                                'mae_dur': nn_losses.mae_dur})

    def recreate(self, path):
        """

        :param path:
        :return:
        """
        path = Path(path)
        with open(str(path / 'MyNN.p'), 'rb') as dump_file:
            d = pickle.load(dump_file)
            model_id = d['model_id']
            input_param = d['input_param']
            opt_param = d['input_param']
            type_loss = d['type_loss']
            step_length = d['step_length']
            model_options = d['model_options']

        self.new_model(model_id=model_id, input_param=input_param, opt_param=opt_param, type_loss=type_loss,
                       step_length=step_length, model_options=model_options)

    def load_weights(self, path):
        """

        :param path:
        :return:
        """
        path = Path(path)
        with open((path / 'weights.p'), 'rb') as dump_file:
            d = pickle.load(dump_file)
            self.model.set_weights(d['weights'])

    def generate(self, input):
        """

        :param input:
        :return:
        """
        return self.model.predict(input, verbose=0)

    def test_function(self, inputs=None, truth=None, generator=None, learning_phase=0):
        self.model._make_test_function()
        if generator is not None:
            # All Midi have to be in same shape.
            bar = progressbar.ProgressBar(maxval=len(generator),
                                          widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage(), ' ',
                                                   progressbar.ETA()])
            bar.start()  # To see it working
            batch = len(generator[0][1][0])
            nb_instruments = len(generator[0][1])
            print('batch', batch, 'instruments', nb_instruments, 'shape', generator[0][0][0].shape)
            ins = generator[0][0] + generator[0][1] + [np.ones((3, 1)), np.ones((1,))] + [
                learning_phase]  # + [np.ones((batch, )) for isnt in range(nb_instruments)] + [learning_phase]
            print('ins', len(ins))
            outputs = self.model.test_function(ins)
            for i in range(1, len(generator)):
                ins = generator[i][0] + generator[i][1] + [np.ones((batch,)) for inst in range(nb_instruments)] + [
                    learning_phase]
                _outputs = self.model.test_function(ins)
                for j in range(len(outputs)):
                    outputs[j] += _outputs[j]
                bar.update(i)
            bar.finish()
            for j in range(len(outputs)):
                outputs[j] /= len(generator)
        else:
            if type(inputs) is not list:
                inputs = [inputs]
            if type(truth) is not list:
                truth = [truth]
            nb_instruments = len(truth)
            batch = len(truth[0])
            ins = inputs + truth + [np.ones(batch) for inst in range(nb_instruments)] + [learning_phase]
            outputs = self.model.test_function(ins)
        return outputs

    def predict_function(self, inputs, learning_phase=0):
        if self.model.predict_function is None:
            self.model._make_predict_function()
        outputs = self.model.predict_function([
            inputs,
            learning_phase
        ])
        return outputs

    @staticmethod
    def allow_growth():
        """

        :return:
        """
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.compat.v1.Session(config=config)
        tf.compat.v1.keras.backend.set_session(sess)

    @staticmethod
    def choose_gpu(gpu):
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu

    @staticmethod
    def clear_session():
        K.clear_session()
