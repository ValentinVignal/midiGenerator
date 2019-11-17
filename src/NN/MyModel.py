import os
from pathlib import Path
import pickle
import numpy as np
import progressbar
from termcolor import colored, cprint
import math
import random

from src.NN.KerasNeuralNetwork import KerasNeuralNetwork
from src.NN.sequences.AllInstSequence import AllInstSequence
from src.NN.sequences.MissingInstSequence import MissingInstSequence
from src.NN.sequences.KerasSequence import KerasSequence
import src.midi.create as midi_create
import src.image.pianoroll as pianoroll
import src.text.summary as summary
import src.global_variables as g


class MyModel:
    """

    """

    def __init__(self, name='name', data=None):
        """

        :param name: The name of the model
        :param work_on: either 'beat' or 'measure'
        :param data: if not None, load the data
        """
        # ----- General -----
        self.total_epochs = 0
        self.name = name
        self.model_id = ''  # Id of the model used
        self.full_name = ''  # Id of this MyModel instance
        self.work_on = None
        self.get_new_full_name()

        self.saved_model_pathlib = Path(
            os.path.join('saved_models', self.full_name))  # Where to saved the trained model

        # ----- Data -----
        self.data_transformed_pathlib = None

        self.instruments = None  # List of instruments used
        self.notes_range = None

        # ----- MySequence -----
        self.my_sequence = None  # Instance of MySequence Generator
        self.batch = None  # Size if the batch
        self.mono = None  # If this is not polyphonic instrument and no rest

        # ----- Neural Network -----
        self.input_param = None  # The parameters for the neural network
        self.keras_nn = None  # Our neural network
        self.train_history = None

        # ------ save_midi_path -----
        self.save_midis_pathlib = None  # Where to save the generated midi files

        if data is not None:
            self.load_data(data)

    # --------------------------------------------------
    #               Class Methods
    # --------------------------------------------------

    @classmethod
    def from_model(cls, id, name='name', data=None):
        myModel = cls(name=name, data=data)
        myModel.load_model(id=id)
        return myModel

    @classmethod
    def with_new_model(cls, model_infos, name='name', work_on=g.work_on, data=None):
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
            work_on=work_on,
            opt_param=get_value('opt_param'),
        )
        return myModel

    @classmethod
    def with_model(cls, id, with_weights=True):
        my_model = cls()
        my_model.recreate_model(id=id, with_weigths=with_weights)
        return my_model

    # --------------------------------------------------
    #                   Properties
    # --------------------------------------------------

    @property
    def nb_steps(self):
        """

        :return:
        """
        return int(self.model_id.split(',')[2])

    @property
    def step_length(self):
        return g.work_on2nb(self.work_on)

    # --------------------------------------------------
    #                   Names function
    # --------------------------------------------------

    def get_full_name(self, i):
        """

        :param i: index
        :return: set up the full name and the path to save the trained model
        """
        full_name = '{0}-m({1})-wo({4})-e({2})-({3})'.format(self.name, self.model_id, self.total_epochs, i,
                                                             g.work_on2letter(self.work_on))
        saved_model_pathlib = Path(os.path.join('saved_models', full_name))
        self.full_name = full_name
        self.saved_model_pathlib = saved_model_pathlib
        print('Get full_name :', colored(self.full_name, 'blue'))

    def get_new_full_name(self):
        """

        :return: set up a new unique full name and the corresponding path to save the trained model
        """
        i = 0
        full_name = '{0}-m({1})-wo({4})-e({2})-({3})'.format(self.name, self.model_id, self.total_epochs, i,
                                                             g.work_on2letter(self.work_on))
        saved_model_pathlib = Path(os.path.join('saved_models', full_name))
        while saved_model_pathlib.exists():
            i += 1
            full_name = '{0}-m({1})-wo({4})-e({2})-({3})'.format(self.name, self.model_id, self.total_epochs, i,
                                                                 g.work_on2letter(self.work_on))
            saved_model_pathlib = Path(os.path.join('saved_models', full_name))
        self.saved_model_pathlib = saved_model_pathlib
        self.full_name = full_name
        print('Got new full_name :', colored(self.full_name, 'blue'))

        self.save_midis_pathlib = None

    def set_name(self, name=None):
        """

        :param name:
        :return:
        """
        self.name = self.name if name is None else name
        self.get_new_full_name()

    # --------------------------------------------------

    def load_data(self, data_transformed_path=None):
        """

        :return: load the data
        """
        self.data_transformed_pathlib = Path(
            data_transformed_path) if data_transformed_path is not None else self.data_transformed_pathlib
        self.input_param = {}
        with open(str(self.data_transformed_pathlib / 'infos_dataset.p'), 'rb') as dump_file:
            d = pickle.load(dump_file)
            self.input_param['input_size'] = d['input_size']
            self.input_param['nb_instruments'] = d['nb_instruments']
            self.instruments = d['instruments']
            self.notes_range = d['notes_range']
            self.mono = d['mono']
        print('data at', colored(data_transformed_path, 'grey', 'on_white'), 'loaded')

    def change_batch_size(self, batch_size):
        if self.my_sequence is not None and self.batch != batch_size:
            self.batch = batch_size
            self.my_sequence.change_batch_size(batch_size=batch_size)

    # --------------------------------------------------
    #               to create or load a NN
    # --------------------------------------------------

    def new_nn_model(self, model_id, work_on=None, opt_param=None, type_loss=g.type_loss, model_options=None,
                     print_model=True):
        """

        :param model_id: modelName;modelParam;nbSteps
        :param work_on:
        :param opt_param:
        :param type_loss:
        :param model_options:
        :param print_model:
        :return: set up the neural network
        """
        try:
            _ = self.input_param['input_size']
            _ = self.input_param['nb_instruments']
        except KeyError:
            print('Load the data before creating a new model')

        self.model_id = model_id
        self.total_epochs = 0
        if work_on is None:
            self.work_on = g.work_on if self.work_on is None else self.work_on
        else:
            self.work_on = work_on

        step_length = g.work_on2nb(self.work_on)
        self.get_new_full_name()

        opt_param = {'lr': g.lr, 'name': 'adam'} if opt_param is None else opt_param

        self.keras_nn = KerasNeuralNetwork()
        self.keras_nn.new_model(model_id=self.model_id,
                                step_length=step_length,
                                input_param=self.input_param,
                                opt_param=opt_param,
                                type_loss=type_loss,
                                model_options=model_options)
        if print_model:
            self.print_model()

    def load_model(self, id, keep_name=True):
        # TODO: only create and load weights
        """

        :param id: id of the model to load
        :param keep_name: if true keep the name, if not, get a new index at the and of the full name
        :return: load a model
        """
        self.name, self.model_id, work_on_letter, total_epochs, indice = id.split('-')
        self.work_on = g.letter2work_on(work_on_letter)
        self.total_epochs = int(total_epochs)
        if keep_name:
            self.get_full_name(indice)
        else:
            self.get_new_full_name()
        path_to_load = Path('saved_models',
                            '{0}-m({1})-wo({4})-e({2})-({3})'.format(self.name, self.model_id, self.total_epochs,
                                                                     indice, work_on_letter))
        self.keras_nn = KerasNeuralNetwork()
        self.keras_nn.load(str(path_to_load / 'MyNN'))
        with open(str(path_to_load / 'infos.p'), 'rb') as dump_file:
            d = pickle.load(dump_file)
            self.input_param = d['nn']['input_param']
            self.instruments = d['instruments']
            self.notes_range = d['notes_range']
            self.work_on = d['work_on']
        self.print_model()
        print('Model', colored(id, 'white', 'on_blue'), 'loaded')

    def recreate_model(self, id, with_weigths=True, print_model=True):
        # TODO: only create and load weights
        """
        create a new model witht the same options as the saved model and then load the weights (if with_weights==True)
        :param id:
        :param with_weigths: if we have to load the weight of the model
        :param print_model:
        :return:
        """
        self.name, self.model_id, work_on_letter, total_epochs, indice = id.split('-')
        self.work_on = g.letter2work_on(work_on_letter)
        self.get_full_name(indice)
        path_to_load = Path('saved_models',
                            '{0}-m({1})-wo({4})-e({2})-({3})'.format(self.name, self.model_id, total_epochs,
                                                                     indice, work_on_letter))
        with open(str(path_to_load / 'infos.p'), 'rb') as dump_file:
            d = pickle.load(dump_file)
            self.input_param = d['nn']['input_param']
            self.instruments = d['instruments']
            self.notes_range = d['notes_range']
            self.work_on = d['work_on']
            self.data_transformed_pathlib = d['data_transformed_pathlib']

        self.keras_nn = KerasNeuralNetwork()
        self.keras_nn.recreate((path_to_load / 'MyNN').as_posix())

        if with_weigths:
            self.keras_nn.load_weights((path_to_load / 'MyNN').as_posix())
            self.total_epochs = int(total_epochs)
            self.get_full_name(indice)
        if print_model:
            self.print_model()

    def load_weights(self, id, keep_name=True):
        # TODO: only create and load weights
        """

        :param id: id of the model to load
        :param keep_name: if true keep the name, if not, get a new index at the and of the full name
        :return: load the weights of a model
        """
        self.name, self.model_id, work_on_letter, total_epochs, indice = id.split('-')
        self.work_on = g.letter2work_on(work_on_letter)
        self.total_epochs = int(total_epochs)
        if keep_name:
            self.get_full_name(indice)
        else:
            self.get_new_full_name()
        path_to_load = Path('saved_models',
                            '{0}-m({1})-wo({4})-e({2})-({3})'.format(self.name, self.model_id, self.total_epochs,
                                                                     indice, work_on_letter))
        self.keras_nn.load_weights(str(path_to_load / 'MyNN.h5'))
        self.print_model()
        print('Weights of the', colored('id', 'white', 'on_blue'), 'model loaded')

    # --------------------------------------------------

    def print_model(self):
        print(self.keras_nn.model.summary())

    # --------------------------------------------------
    #                Train the model
    # --------------------------------------------------

    def train(self, epochs=None, batch=None, callbacks=[], verbose=1, noise=g.noise, validation=0.0):
        """

        :param epochs:
        :param batch:
        :param callbacks:
        :param verbose:
        :param noise:
        :param validation:
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
            self.my_sequence = MissingInstSequence(
                path=str(self.data_transformed_pathlib),
                nb_steps=int(self.model_id.split(',')[2]),
                batch_size=self.batch,
                work_on=self.work_on
            )
        if noise is not None:
            self.my_sequence.set_noise(noise)

        # Actual train
        print(colored('Training...', 'blue'))
        self.train_history = self.keras_nn.train_seq(epochs=epochs, generator=self.my_sequence, callbacks=callbacks,
                                                     verbose=verbose, validation=validation)

        # Update parameters
        self.total_epochs += epochs
        self.get_new_full_name()
        print(colored('Training done', 'green'))

    # --------------------------------------------------
    #                Test the model
    # --------------------------------------------------

    def evaluate(self, batch=None):
        if batch is not None:
            self.batch = batch
        if self.batch is None:
            self.batch = 4
        cprint('Evaluation', 'blue')
        if self.my_sequence is None:
            self.my_sequence = AllInstSequence(
                path=str(self.data_transformed_pathlib),
                nb_steps=int(self.model_id.split(',')[2]),
                batch_size=self.batch,
                work_on=self.work_on
            )
        evaluation = self.keras_nn.evaluate(generator=self.my_sequence)

        metrics_names = self.keras_nn.model.metrics_names
        text = ''
        for i in range(len(metrics_names)):
            text += metrics_names[i] + ' ' + colored(evaluation[i], 'magenta') + ' -- '
        print(text)

    def test_on_batch(self, i=0, batch_size=4):
        self.my_sequence.change_batch_size(batch_size=batch_size)
        x, y = self.my_sequence[i]
        evaluation = self.keras_nn.model.test_on_batch(x=x, y=y, sample_weight=None)

        metrics_names = self.keras_nn.model.metrics_names
        text = ''
        for i in range(len(metrics_names)):
            text += metrics_names[i] + ' ' + colored(evaluation[i], 'magenta') + ' -- '
        print(text)

    def predict_on_batch(self, i, batch_size=4):
        self.my_sequence.change_batch_size(batch_size=batch_size)
        x, y = self.my_sequence[i]
        evaluation = self.keras_nn.model.predict_on_batch(x=x)

        return evaluation

    def compare_test_predict_on_batch(self, i, batch_size=4):
        print('compare test predict on batch')
        self.test_on_batch(i, batch_size=batch_size)
        x, yt = self.my_sequence[i]
        yp = self.predict_on_batch(i, batch_size=batch_size)
        pianoroll.see_compare_on_batch(x, yt, yp)

    # --------------------------------------------------
    #                Save the model
    # --------------------------------------------------

    def save_model(self, path=None):
        # TODO: Only save weights and information
        """

        :param path: path were to save the model, if Nonem it will be at self.saved_model_path
        :return:
        """
        path_to_save = self.saved_model_pathlib if path is None else Path(path)
        path_to_save.mkdir(parents=True, exist_ok=True)  # Creation of this folder
        self.saved_model_pathlib.mkdir(parents=True, exist_ok=True)
        self.keras_nn.save(str(path_to_save / 'MyNN'))
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
                'notes_range': self.notes_range,
                'work_on': self.work_on,
                'data_transformed_pathlib': self.data_transformed_pathlib,
                'mono': self.mono
            }, dump_file)
        summary.summarize_train(path_to_save, **{
            'full_name': self.full_name,
            'epochs': self.total_epochs,
            'input_param': self.input_param,
            'instruments': self.instruments,
            'notes_range': self.notes_range,
            'work_on': self.work_on
        })

        # TODO: Uncomment and make it work when there is a accuracy
        """
        if self.mono:
            summary.save_train_history_mono(self.train_history, len(self.instruments), path_to_save)
        else:
            summary.save_train_history(self.train_history, len(self.instruments), path_to_save)
        """

        print(colored(f'Model saved in {path_to_save}', 'green'))
        return path_to_save

    # --------------------------------------------------

    def print_weights(self):
        """
        Print the weights
        :return:
        """
        for layer in self.keras_nn.model.layers:
            lstm_weights = layer.get_weights()  # list of numpy arrays
            print('Lstm weights:', lstm_weights)

    # --------------------------------------------------

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
        print('new save path for midi files :', colored(str(self.save_midis_pathlib), 'cyan'))

    # --------------------------------------------------
    #                   To generate
    # --------------------------------------------------

    def generate_fom_data(self, nb_seeds=10, new_data_path=None, length=None, new_save_path=None, save_images=False,
                          no_duration=False, verbose=1):
        """
        Generate midi file from the seed and the trained model
        :param nb_seeds: number of seeds for the generation
        :param new_data_path: The path of the seed
        :param length: Length of th generation
        :param new_save_path:
        :param save_images: To save the pianoroll of the generation (.jpg images)
        :param no_duration: if True : all notes will be the shortest length possible
        :param verbose: Level of verbose
        :return:
        """
        nb_steps = int(self.model_id.split(',')[2])

        # ---------- Verify the inputs ----------

        # ----- Create the seed -----
        need_new_sequence = False
        if (new_data_path is not None) and (new_data_path != self.data_transformed_pathlib.as_posix()):
            self.load_data(new_data_path)
            need_new_sequence = True
        if self.data_transformed_pathlib is None:
            raise Exception('Some data need to be loaded before generating')
        if self.my_sequence is None:
            need_new_sequence = True
        if need_new_sequence:
            self.my_sequence = AllInstSequence(
                path=str(self.data_transformed_pathlib),
                nb_steps=nb_steps,
                batch_size=1,
                work_on=self.work_on)
        else:
            self.my_sequence.change_batch_size(1)

        seeds_indexes = random.sample(range(len(self.my_sequence)), nb_seeds)

        step_length = g.work_on2nb(self.work_on)
        # -- Length --
        length = length if length is not None else 200
        # -- For save midi path --
        if type(new_save_path) is str or (
                type(new_save_path) is bool and new_save_path) or (
                new_save_path is None and self.save_midis_pathlib is None):
            self.get_new_save_midis_path(path=new_save_path)
        # --- Done Verifying the inputs ---

        self.save_midis_pathlib.mkdir(parents=True, exist_ok=True)
        cprint('Start generating ...', 'blue')
        for s in range(nb_seeds):
            cprint('Generation {0}/{1}'.format(s + 1, nb_seeds), 'blue')
            generated = np.array(
                self.my_sequence[seeds_indexes[s]][0])  # (nb_instruments, 1, nb_steps, step_size, inputs_size, 2)
            bar = progressbar.ProgressBar(maxval=length,
                                          widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage(), ' ',
                                                   progressbar.ETA()])
            bar.start()  # To see it working
            for l in range(length):
                samples = generated[:, :, l:]  # (nb_instruments, 1, nb_steps, length, 88, 2)   # 1 = batch
                # expanded_samples = np.expand_dims(samples, axis=0)
                preds = self.keras_nn.generate(
                    input=list(samples))  # (nb_instruments, batch=1 , nb_steps=1, length, 88, 2)
                preds = np.asarray(preds).astype('float64')  # (nb_instruments, 1, 1, step_size, input_size, 2)
                if len(preds.shape) == 4:  # Only one instrument : output of nn not a list
                    preds = preds[np.newaxis]
                next_array = midi_create.normalize_activation(preds)  # Normalize the activation part
                generated = np.concatenate((generated, next_array), axis=2)  # (nb_instruments, nb_steps, length, 88, 2)

                bar.update(l + 1)
            bar.finish()

            generated_midi_final = np.reshape(generated, (
                generated.shape[0], generated.shape[2] * generated.shape[3], generated.shape[4],
                generated.shape[5]))  # (nb_instruments, nb_step * length, 88 , 2)
            generated_midi_final = np.transpose(generated_midi_final,
                                                (0, 2, 1, 3))  # (nb_instruments, 88, nb_steps * length, 2)
            output_notes_list = midi_create.matrix_to_midi(generated_midi_final, instruments=self.instruments,
                                                           notes_range=self.notes_range, no_duration=no_duration,
                                                           mono=self.mono)

            # --- find the name for the midi_file ---
            i = 0
            m_str = "lstm_out_({0}).mid".format(i)
            while (self.save_midis_pathlib / m_str).exists():
                i += 1
                m_str = "lstm_out_({0}).mid".format(i)
            path_to_save = str(self.save_midis_pathlib / m_str)
            path_to_save_img = str(self.save_midis_pathlib / 'lstm_out_({0}).jpg'.format(i))

            midi_create.print_informations(nb_steps=nb_steps * step_length, matrix=generated_midi_final,
                                           notes_list=output_notes_list, verbose=verbose)

            # Saving the midi file
            midi_create.save_midi(output_notes_list=output_notes_list, instruments=self.instruments,
                                  path=path_to_save, )
            if save_images:
                pianoroll.save_pianoroll(array=generated_midi_final,
                                         path=path_to_save_img,
                                         seed_length=nb_steps * step_length,
                                         instruments=self.instruments,
                                         mono=self.mono)

        if self.batch is not None:
            self.my_sequence.change_batch_size(self.batch)

        summary.summarize_generation(str(self.save_midis_pathlib), **{
            'full_name': self.full_name,
            'epochs': self.total_epochs,
            'input_param': self.input_param,
            'instruments': self.instruments,
            'notes_range': self.notes_range
        })

        cprint('Done Generating', 'green')

    def fill_instruments(self, max_length=None, no_duration=False, verbose=1):
        """

        :param max_length:
        :param no_duration:
        :param verbose:
        :return:
        """
        # ----- Parameters -----
        max_length = 300 / g.work_on2nb(self.work_on) if max_length is None else max_length

        # ----- Variables -----
        if self.data_transformed_pathlib is None:
            raise Exception('Some data need to be loaded before comparing the generation')
        sequence = KerasSequence(
            path=self.data_transformed_pathlib,
            nb_steps=self.nb_steps,
            batch_size=1,
            work_on=self.work_on
        )  # Return array instead of list (for instruments)
        max_length = int(min(max_length, len(sequence)))
        nb_instruments = sequence.nb_instruments
        # ----- Seeds -----
        truth = sequence[0][0]
        filled_list = [np.copy(truth) for inst in range(nb_instruments)]
        for inst in range(nb_instruments):
            filled_list[inst][inst] = np.nan

        # ----- Generation -----
        bar = progressbar.ProgressBar(maxval=max_length,
                                      widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage(), ' ',
                                               progressbar.ETA()])
        bar.start()  # To see it working
        for l in range(max_length):
            s_input, s_output = sequence[l]
            to_fill_list = [np.copy(s_input) for inst in range(nb_instruments)]
            for inst in range(nb_instruments):
                to_fill_list[inst][inst] = np.nan
            nn_input = np.concatenate(
                (*to_fill_list,),
                axis=1
            )  # (nb_instruments, batch=nb_instruments, nb_steps, step_size, input_size, channels)
            preds = self.keras_nn.generate(input=list(nn_input))

            preds = np.asarray(preds).astype(
                'float64')  # (nb_instruments, bath=nb_instruments, nb_steps=1, step_size, input_size, channels)
            if len(preds.shape) == 5:  # Only one instrument : output of nn not a list
                preds = np.expand_dims(preds, axis=0)
            if len(s_output.shape) == 5:  # Only one instrument : output of nn not a list
                s_output = np.expand_dims(s_output)
            truth = np.concatenate((truth, s_output), axis=2)
            for inst in range(nb_instruments):
                p = np.copy(s_output)
                p[inst] = np.take(preds, axis=1, indices=[inst])[inst]
                filled_list[inst] = np.concatenate(
                    (filled_list[inst], p),
                    axis=2)  # (nb_instruments, batch=1, nb_steps, step_size, input_size, channels)
            bar.update(l + 1)
        bar.finish()

        # -------------------- Compute notes list --------------------
        # ----- Reshape -----
        truth = np.reshape(truth, (truth.shape[0], truth.shape[2] * truth.shape[3],
                                   *truth.shape[
                                    4:]))  # (nb_instruments, nb_steps * step_size, input_size, channels)
        truth = np.transpose(truth, axes=(0, 2, 1, 3))  # (nb_instruments, input_size, length, channels)
        for inst in range(nb_instruments):
            s = filled_list[inst].shape
            filled_list[inst] = np.reshape(filled_list[inst], (
                s[0], s[2] * s[3], *s[4:]))  # (nb_instruments, nb_steps * step_size, input_size, channels)
            filled_list[inst] = np.transpose(filled_list[inst],
                                             axes=(0, 2, 1, 3))  # (nb_instruments, input_size, length, channels)
        # ----- Notes -----
        output_notes_truth = midi_create.matrix_to_midi(truth,
                                                        instruments=self.instruments,
                                                        notes_range=self.notes_range, no_duration=no_duration,
                                                        mono=self.mono)
        output_notes_inst = [
            midi_create.matrix_to_midi(filled_list[inst], instruments=self.instruments,
                                       notes_range=self.notes_range,
                                       no_duration=no_duration, mono=self.mono)
            for inst in range(nb_instruments)
        ]

        # ---------- find the name for the midi_file ----------
        self.get_new_save_midis_path()
        self.save_midis_pathlib.mkdir(parents=True, exist_ok=True)

        # -------------------- Save Results --------------------
        # Truth
        midi_create.print_informations(nb_steps=self.nb_steps * self.step_length,
                                       matrix=truth, notes_list=output_notes_truth, verbose=verbose)
        # TODO: Accuracy
        midi_create.save_midi(output_notes_list=output_notes_truth, instruments=self.instruments,
                              path=self.save_midis_pathlib / 'truth.mid')
        pianoroll.save_pianoroll(array=truth,
                                 path=self.save_midis_pathlib / 'truth.jpg',
                                 seed_length=self.nb_steps * self.step_length,
                                 instruments=self.instruments,
                                 mono=self.mono)
        # Missing instruments
        for inst in range(nb_instruments):
            midi_create.print_informations(nb_steps=self.nb_steps * self.step_length,
                                           matrix=filled_list[inst], notes_list=output_notes_inst[inst],
                                           verbose=verbose)
            # TODO: Accuracy
            midi_create.save_midi(output_notes_list=output_notes_inst[inst], instruments=self.instruments,
                                  path=self.save_midis_pathlib / f'missing_{inst}.mid')
            pianoroll.save_pianoroll(array=filled_list[inst],
                                     path=self.save_midis_pathlib / f'missing_{inst}.jpg',
                                     seed_length=self.nb_steps * self.step_length,
                                     instruments=self.instruments,
                                     mono=self.mono)

    def compare_generation(self, max_length=None, no_duration=False, verbose=1):
        """

        :return:
        """
        # -------------------- Find informations --------------------
        if self.data_transformed_pathlib is None:
            raise Exception('Some data need to be loaded before comparing the generation')
        nb_steps = int(self.model_id.split(',')[2])
        if self.my_sequence is None:
            self.my_sequence = AllInstSequence(
                path=str(self.data_transformed_pathlib),
                nb_steps=nb_steps,
                batch_size=1,
                work_on=self.work_on)
        else:
            self.my_sequence.change_batch_size(1)
        self.my_sequence.set_noise(0)
        max_length = len(self.my_sequence) if max_length is None else min(max_length, len(self.my_sequence))

        # -------------------- Construct seeds --------------------
        generated = np.array(self.my_sequence[0][0])  # (nb_instrument, 1, nb_steps, step_size, input_size, 2) (1=batch)
        generated_helped = np.copy(generated)  # Each step will take the truth as an input
        generated_truth = np.copy(generated)  # The truth

        # -------------------- Generation --------------------
        bar = progressbar.ProgressBar(maxval=max_length,
                                      widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage(), ' ',
                                               progressbar.ETA()])
        bar.start()  # To see it working
        for l in range(max_length):
            ms_input, ms_output = self.my_sequence[l]
            sample = np.concatenate((generated[:, :, l:], np.array(ms_input)),
                                    axis=1)  # (nb_instruments, 2, nb_steps, step_size, input_size, 2)

            # Generation
            preds = self.keras_nn.generate(input=list(sample))

            # Reshape
            preds = np.asarray(preds).astype('float64')  # (nb_instruments, batch=2, nb_steps=1, length, 88, 2)
            preds_truth = np.array(ms_output)  # (nb_instruments, 1, 1, step_size, input_size, 2)
            # if only one instrument
            if len(preds.shape) == 5:  # Only one instrument : output of nn not a list
                preds = np.expand_dims(preds, axis=0)
            if len(preds_truth.shape) == 5:  # Only one instrument : output of nn not a list
                preds_truth = np.expand_dims(preds_truth)
            preds = midi_create.normalize_activation(preds, mono=self.mono)  # Normalize the activation part
            preds_helped = preds[:, [1]]  # (nb_instruments, 1, 1, length, 88, 2)
            preds = preds[:, [0]]

            # Concatenation
            generated = np.concatenate((generated, preds), axis=2)  # (nb_instruments, 1, nb_steps, length, 88, 2)
            generated_helped = np.concatenate((generated_helped, preds_helped),
                                              axis=2)  # (nb_instruments, 1, nb_steps, length, 88, 2)
            generated_truth = np.concatenate((generated_truth, preds_truth), axis=2)
            bar.update(l + 1)
        bar.finish()

        # -------------------- Compute notes list --------------------
        # Generated
        generated_midi_final = np.reshape(generated, (
            generated.shape[0], generated.shape[2] * generated.shape[3], generated.shape[4],
            generated.shape[5]))  # (nb_instruments, nb_step * length, 88 , 2)
        generated_midi_final = np.transpose(generated_midi_final,
                                            (0, 2, 1, 3))  # (nb_instruments, 88, nb_steps * length, 2)
        output_notes_list = midi_create.matrix_to_midi(generated_midi_final, instruments=self.instruments,
                                                       notes_range=self.notes_range, no_duration=no_duration,
                                                       mono=self.mono)
        # Helped
        generated_midi_final_helped = np.reshape(generated_helped, (
            generated_helped.shape[0], generated_helped.shape[2] * generated_helped.shape[3],
            generated_helped.shape[4], generated_helped.shape[5]))  # (nb_instruments, nb_step * length, 88 , 2)
        generated_midi_final_helped = np.transpose(generated_midi_final_helped,
                                                   (0, 2, 1, 3))  # (nb_instruments, 88, nb_steps * length, 2)
        output_notes_list_helped = midi_create.matrix_to_midi(generated_midi_final_helped, instruments=self.instruments,
                                                              notes_range=self.notes_range, no_duration=no_duration,
                                                              mono=self.mono)
        # Truth
        generated_midi_final_truth = np.reshape(generated_truth, (
            generated_truth.shape[0], generated_truth.shape[2] * generated_truth.shape[3], generated_truth.shape[4],
            generated_truth.shape[5]))  # (nb_instruments, nb_step * length, 88 , 2)
        generated_midi_final_truth = np.transpose(generated_midi_final_truth,
                                                  (0, 2, 1, 3))  # (nb_instruments, 88, nb_steps * length, 2)
        output_notes_list_truth = midi_create.matrix_to_midi(generated_midi_final_truth,
                                                             instruments=self.instruments,
                                                             notes_range=self.notes_range, no_duration=no_duration,
                                                             mono=self.mono)

        # ---------- find the name for the midi_file ----------
        self.get_new_save_midis_path()
        self.save_midis_pathlib.mkdir(parents=True, exist_ok=True)

        # -------------------- Save Results --------------------
        if self.work_on == 'note':
            step_length = 1
        elif self.work_on == 'beat':
            step_length = g.step_per_beat
        elif self.work_on == 'measure':
            step_length = 4 * g.step_per_beat
        # -- Generated --
        path_to_save = str(self.save_midis_pathlib / 'generated.mid')
        path_to_save_img = str(self.save_midis_pathlib / 'generated.jpg')
        midi_create.print_informations(nb_steps=nb_steps * step_length, matrix=generated_midi_final,
                                       notes_list=output_notes_list, verbose=verbose)
        # Print the accuracy
        if self.mono:
            argmax = np.argmax(generated_midi_final, axis=1)
            argmax_truth = np.argmax(generated_midi_final_truth, axis=1)
            accuracies = [(np.count_nonzero(
                argmax[i, nb_steps * step_length:] == argmax_truth[i,
                                                      nb_steps * step_length:]) / argmax[i,
                                                                                  nb_steps * step_length:].size)
                          for i in range(len(self.instruments))]
        else:
            accuracies = [((np.count_nonzero(
                generated_midi_final[i, :, nb_steps * step_length:, 0] == generated_midi_final_truth[i, :,
                                                                          nb_steps * step_length:,
                                                                          0])) / (generated_midi_final[i, :,
                                                                                  nb_steps * step_length:, 0].size)) for
                          i
                          in
                          range(len(self.instruments))]
        accuracy = sum(accuracies) / len(accuracies)
        print('Accuracy of the generation :', colored(accuracies, 'magenta'), ', overall :',
              colored(accuracy, 'magenta'))

        # Saving the midi file
        midi_create.save_midi(output_notes_list=output_notes_list, instruments=self.instruments,
                              path=path_to_save, )
        pianoroll.save_pianoroll(array=generated_midi_final,
                                 path=path_to_save_img,
                                 seed_length=nb_steps * step_length,
                                 instruments=self.instruments,
                                 mono=self.mono)

        # -- Helped --
        path_to_save = str(self.save_midis_pathlib / 'generated_helped.mid')
        path_to_save_img = str(self.save_midis_pathlib / 'generated_helped.jpg')
        midi_create.print_informations(nb_steps=nb_steps * step_length, matrix=generated_midi_final_helped,
                                       notes_list=output_notes_list_helped, verbose=verbose)
        # Print the accuracy
        if self.mono:
            argmax_helped = np.argmax(generated_midi_final_helped, axis=1)
            argmax_truth = np.argmax(generated_midi_final_truth, axis=1)
            accuracies_helped = [(np.count_nonzero(
                argmax_helped[i, nb_steps * step_length:] == argmax_truth[i,
                                                             nb_steps * step_length:]) / argmax_helped[i,
                                                                                         nb_steps * step_length:].size)
                                 for i in range(len(self.instruments))]
        else:
            accuracies_helped = [(np.count_nonzero(
                generated_midi_final_helped[i, :, nb_steps * step_length:, 0] == generated_midi_final_truth[i, :,
                                                                                 nb_steps * step_length:,
                                                                                 0]) / generated_midi_final_helped[i, :,
                                                                                       nb_steps * step_length:, 0].size)
                                 for
                                 i in
                                 range(len(self.instruments))]
        accuracy_helped = sum(accuracies_helped) / len(accuracies_helped)

        print('Accuracy of the generation helped :', colored(accuracies_helped, 'magenta'), ', overall :',
              colored(accuracy_helped, 'magenta'))

        # Saving the midi file
        midi_create.save_midi(output_notes_list=output_notes_list_helped, instruments=self.instruments,
                              path=path_to_save, )
        pianoroll.save_pianoroll(array=generated_midi_final_helped,
                                 path=path_to_save_img,
                                 seed_length=nb_steps * step_length,
                                 instruments=self.instruments,
                                 mono=self.mono)

        # -- Truth --
        path_to_save = str(self.save_midis_pathlib / 'generated_truth.mid')
        path_to_save_img = str(self.save_midis_pathlib / 'generated_truth.jpg')
        midi_create.print_informations(nb_steps=nb_steps * step_length, matrix=generated_midi_final_truth,
                                       notes_list=output_notes_list_truth, verbose=verbose)

        # Saving the midi file
        midi_create.save_midi(output_notes_list=output_notes_list_truth, instruments=self.instruments,
                              path=path_to_save, )
        pianoroll.save_pianoroll(array=generated_midi_final_truth,
                                 path=path_to_save_img,
                                 seed_length=nb_steps * step_length,
                                 instruments=self.instruments,
                                 mono=self.mono)

        text = 'Generated :\n\tAccuracy : {0}, Accuracies : {1}\n'.format(accuracy, accuracies)
        text += 'Generated Helped :\n\tAccuracy : {0}, Accuracies : {1}'.format(accuracy_helped, accuracies_helped)
        with open(str(self.save_midis_pathlib / 'Results.txt'), 'w') as f:
            f.write(text)

        cprint('Done Generating', 'green')
