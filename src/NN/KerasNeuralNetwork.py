import os
import tensorflow as tf
from pathlib import Path
import pickle
import dill
import json
import math
from time import time
import gc

import src.global_variables as g
from src.NN import Sequences
from . import Models
from src import tb

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
        self.opt_param = None
        self.loss_options = None
        self.decay = None
        self.callbacks = []

        self.model_options = None

        # Spare GPU
        # tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        self.allow_growth()

        log_dir = os.path.join('tensorboard', f'{time()}')
        self.tensorboard = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir,
            write_graph=True,
            write_images=True,
            # embedding_freq=0.5,
            # write_grad=True
        )

    def __del__(self):
        del self.tensorboard
        del self.model
        del self.callbacks
        del self.model_options
        del self.opt_param
        del self.loss_options

    def new_model(self, model_id, input_param, opt_param, step_length=1, model_options={}, loss_options={}):
        """

        :param loss_options:
        :param model_id: model_name;model_param;nb_steps
        :param input_param:
        :param opt_param: {'lr', 'name'}
        :param step_length:
        :param model_options:
        :return: the neural network
        """

        self.step_length = step_length
        self.loss_options = loss_options

        model_name, model_param_s, nb_steps = model_id.split(g.split_model_id)
        nb_steps = int(nb_steps)

        nn_model = Models.from_name[model_name]

        # Load model param .json file
        json_path = os.path.join('src',
                                 'NN',
                                 'Models',
                                 model_name,
                                 '{0}.json'.format(model_param_s))
        with open(json_path) as json_file:
            model_param = json.load(json_file)

        self.opt_param = opt_param
        optimizer, self.decay = KerasNeuralNetwork.create_optimizer(**self.opt_param)

        model_dict = nn_model.create_model(
            input_param=input_param,
            model_param=model_param,
            nb_steps=nb_steps,
            optimizer=optimizer,
            step_length=step_length,
            model_options=model_options,
            loss_options=loss_options
        )
        self.model = model_dict.get('model')
        self.callbacks.extend(model_dict.get('callbacks', []))
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
            'lr': g.lr,
            'decay_drop': g.decay_drop,
            'epochs_drop': g.epochs_drop,
            'decay': g.decay
        }
        opt_param.update(kwargs)

        # ----- Optimizer -----
        optimizer = None
        if opt_param['name'] == 'adam':
            optimizer = tf.keras.optimizers.Adam(lr=opt_param['lr'], beta_1=0.9, beta_2=0.999, epsilon=None,
                                                 amsgrad=False, decay=opt_param['decay'])
        elif opt_param['name'] == 'sgd':
            optimizer = tf.keras.optimizers.SGD(lr=opt_param['lr'], decay=opt_param['decay'])

        # TODO: to use it, make it work with LSTM and no eager execution
        # ----- Decay -----
        step_decay = dill.loads(
            KerasNeuralNetwork.decay_func(lr_init=opt_param['lr'], drop=opt_param['decay_drop'],
                                          epochs_drop=opt_param['epochs_drop']))
        # lrate = tf.keras.callbacks.LearningRateScheduler(step_decay)
        # callback_list = [lrate]

        return optimizer, step_decay

    # TODO: Make it work (not working for now)
    @staticmethod
    def decay_func(lr_init, **kwargs):
        def step_decay(epoch):
            lrate = lr_init * math.pow(kwargs['decay_drop'], math.floor(epoch / kwargs['epochs_drop']))
            return lrate

        return dill.dumps(step_decay)

    def train_seq(self, epochs, generator, callbacks=[], verbose=1, validation=0.0, sequence_to_numpy=False):
        """

        :param sequence_to_numpy: To load all the generator into numpy files to train faster (only for small dataset)
        :param epochs:
        :param generator:
        :param callbacks:
        :param verbose:
        :param validation:
        :return:
        """
        # TODO: To do custom decay: make it work with LSTM and non eager execution
        # callback_list = [tf.keras.callbacks.LearningRateScheduler(self.decay), self.tensorboard] + callbacks
        for callback in self.callbacks:
            callback.update_with_fit_args(epochs=epochs)
        callback_list = self.callbacks + [self.tensorboard] + callbacks

        if sequence_to_numpy:
            print('Loading all the training data as numpy arrays...')
            x, y = Sequences.sequence_to_numpy(sequence=generator)
            history = self.model.fit(x=x, y=y, epochs=epochs, validation_split=validation, shuffle=True,
                                     callbacks=callback_list, batch_size=generator.batch_size)
            del x, y
            gc.collect()

        else:
            if validation > 0:
                generator_train, generator_valid = Sequences.TrainValSequence.get_train_valid_sequence(generator,
                                                                                                       validation_split=validation)

                history = self.model.fit_generator(epochs=epochs, generator=generator_train,
                                                   validation_data=generator_valid,
                                                   shuffle=True, verbose=verbose, callbacks=callback_list)
            else:  # So it won't print a lot of lines for nothing
                history = self.model.fit_generator(epochs=epochs, generator=generator,
                                                   shuffle=True, verbose=verbose, callbacks=callback_list)

        return history.history

    def evaluate(self, generator, verbose=1):
        evaluation = self.model.evaluate_generator(generator=generator, verbose=verbose)
        return evaluation

    def save(self, path):
        """

        :param path:
        :return:
        """
        path = Path(path)
        # shutil.rmtree(path.as_posix(), ignore_errors=True)
        path.mkdir(exist_ok=True, parents=True)
        with open(path / 'MyNN.p', 'wb') as dump_file:
            pickle.dump(
                dict(
                    model_id=self.model_id,
                    input_param=self.input_param,
                    opt_param=self.opt_param,
                    step_length=self.step_length,
                    model_options=self.model_options,
                    loss_options=self.loss_options
                ), dump_file
            )
        self.save_weights(path=path)

        if self.tensorboard is not None:
            # Get the folder where all the Tensorboard information are stored
            log_dir = self.tensorboard.log_dir
            # Get the training data
            train_data = tb.get_tensorboard_data(path=log_dir)
            # Save the plot images
            tb.save_tensorboard_plots(data=train_data, path=path / 'plots')


    def save_weights(self, path):
        """

        :param path:
        :return:
        """
        path = Path(path)
        path.mkdir(exist_ok=True, parents=True)
        with open(path / 'weights.p', 'wb') as dump_file:
            pickle.dump(
                dict(
                    weights=self.model.get_weights()
                ), dump_file
            )

    def recreate(self, path, with_weights=True):
        """

        :param path:
        :return:
        """
        path = Path(path)
        with open(str(path / 'MyNN.p'), 'rb') as dump_file:
            d = pickle.load(dump_file)
            model_id = d['model_id']
            input_param = d['input_param']
            opt_param = d['opt_param']
            step_length = d['step_length']
            model_options = d['model_options']
            loss_options = d['loss_options']

        self.new_model(model_id=model_id, input_param=input_param, opt_param=opt_param, step_length=step_length,
                       model_options=model_options, loss_options=loss_options)
        if with_weights:
            self.load_weights(path=path)

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

    # ------------------------------------------------------------
    #                       Static methods
    # ------------------------------------------------------------

    @staticmethod
    def allow_growth():
        """

        :return:
        """
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")

            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)

    @staticmethod
    def choose_gpu(gpu):
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu

    @staticmethod
    def clear_session():
        K.clear_session()

    @staticmethod
    def disable_eager_exection():
        tf.compat.v1.disable_eager_execution()

