import os
from pathlib import Path
import pickle
import numpy as np
import progressbar
import random
from termcolor import colored, cprint
import math

from src.NN.MyNN import MyNN
from src.NN.data_generator import MySequence, MySequenceBeat
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
        self.get_new_full_name()
        self.work_on = None

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
    def from_model(cls, id, name='name', data=None):
        myModel = cls(name=name, data=data)
        myModel.load_model(id=id)
        return myModel

    @classmethod
    def with_model(cls, model_infos, name='name', work_on=g.work_on, data=None):
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
        print('Get full_name :', colored(self.full_name, 'blue'))

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
        print('Got new full_name :', colored(self.full_name, 'blue'))

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
        print('data at', colored(data_transformed_path, 'grey', 'on_white'), 'loaded')

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
        self.get_new_full_name()

        opt_param = {'lr': g.lr, 'name': 'adam'} if opt_param is None else opt_param

        if work_on is None:
            self.work_on = g.work_on if self.work_on is None else self.work_on
        else:
            self.work_on = work_on
        if self.work_on == 'note':
            step_length = 1
        elif self.work_on == 'beat':
            step_length = g.step_per_beat
        elif self.work_on == 'measure':
            step_length = 4 * g.step_per_beat
        else:
            raise Exception('Work_on type unkown : {0}'.format(work_on))

        self.my_nn = MyNN()
        self.my_nn.new_model(model_id=self.model_id,
                             step_length=step_length,
                             input_param=self.input_param,
                             opt_param=opt_param,
                             type_loss=type_loss,
                             model_options=model_options)
        if print_model:
            self.print_model()

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
            self.notes_range = d['notes_range']
            self.work_on = d['work_on']
        self.print_model()
        print('Model', colored(id, 'white', 'on_blue'), 'loaded')

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
        self.print_model()
        print('Weights of the', colored('id', 'white', 'on_blue'), 'model loaded')

    def print_model(self):
        print(self.my_nn.model.summary())

    def train(self, epochs=None, batch=None, callbacks=[], verbose=1):
        """

        :param epochs:
        :param batch:
        :param callbacks:
        :param verbose:
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
            self.my_sequence = MySequenceBeat(
                path=str(self.data_transformed_pathlib),
                nb_steps=int(self.model_id.split(',')[2]),
                batch_size=self.batch,
                work_on=self.work_on
            )

        # Actual train
        print(colored('Training...', 'blue'))
        self.my_nn.train_seq(epochs=epochs, generator=self.my_sequence, callbacks=callbacks, verbose=verbose)

        # Update parameters
        self.total_epochs += epochs
        self.data_seed_pathlib = self.data_transformed_pathlib
        self.get_new_full_name()
        print(colored('Training done', 'green'))

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
                'notes_range': self.notes_range,
                'work_on': self.work_on
            }, dump_file)
        summary.summarize_train(path_to_save, **{
            'full_name': self.full_name,
            'epochs': self.total_epochs,
            'input_param': self.input_param,
            'instruments': self.instruments,
            'notes_range': self.notes_range,
            'work_on': self.work_on
        })

        print(colored('Model saved in {0}'.format(path_to_save), 'green'))
        return path_to_save

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
        print('new save path for midi files :', colored(str(self.save_midis_pathlib), 'cyan'))

    def generate(self, seed=None, length=None, new_save_path=None, save_images=False, no_duration=False, verbose=1):
        """
        Generate midi file from the seed and the trained model
        :param seed: seed for the generation
        :param length: Length of th generation
        :param new_save_path:
        :param save_images: To save the pianoroll of the generation (.jpg images)
        :param no_duration: if True : all notes will be the shortest length possible
        :param verbose: Level of verbose
        :return:
        """
        # --- Verify the inputs ---
        # -- Nb Steps --
        nb_steps = int(self.model_id.split(',')[2])
        # -- Seed --
        if type(seed) is list:
            pass
        elif seed is None:
            seed = 1
        elif type(seed) is int:
            if self.work_on == 'note':
                step_length = 1
            elif self.work_on == 'beat':
                step_length = g.step_per_beat
            elif self.work_on == 'measure':
                step_length = 4 * g.step_per_beat
            else:
                raise Exception('Unkown work_on type : {0}'.format(self.work_on))
            seed = self.get_seed(nb_steps=nb_steps, step_length=step_length, number=seed)
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
        for s in range(len(seed)):
            cprint('Generation {0}/{1}'.format(s + 1, len(seed)), 'blue')
            generated = seed[s]
            bar = progressbar.ProgressBar(maxval=length,
                                          widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage(), ' ',
                                                   progressbar.ETA()])
            bar.start()  # To see it working
            for l in range(length):
                samples = generated[:, np.newaxis, l:]  # (nb_instruments, 1, nb_steps, length, 88, 2)   # 1 = batch
                # expanded_samples = np.expand_dims(samples, axis=0)
                preds = self.my_nn.generate(input=list(samples))  # (nb_instruments, 1, length, 88, 2)
                preds = np.asarray(preds).astype('float64')  # (nb_instruments, 1, length, 88, 2)
                if len(preds.shape) == 4:  # Only one instrument : output of nn not a list
                    preds = preds[np.newaxis, :, :, :, :]
                next_array = midi_create.normalize_activation(preds)  # Normalize the activation part
                generated = np.concatenate((generated, next_array), axis=1)  # (nb_instruments, nb_steps, length, 88, 2)

                bar.update(l + 1)
            bar.finish()

            generated_midi_final = np.reshape(generated, (
                generated.shape[0], generated.shape[1] * generated.shape[2], generated.shape[3],
                generated.shape[4]))  # (nb_instruments, nb_step * length, 88 , 2)
            generated_midi_final = np.transpose(generated_midi_final,
                                                (0, 2, 1, 3))  # (nb_instruments, 88, nb_steps * length, 2)
            output_notes_list = midi_create.matrix_to_midi(generated_midi_final, instruments=self.instruments,
                                                           notes_range=self.notes_range, no_duration=no_duration)

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
                                         instruments=self.instruments)

        summary.summarize_generation(str(self.save_midis_pathlib), **{
            'full_name': self.full_name,
            'epochs': self.total_epochs,
            'input_param': self.input_param,
            'instruments': self.instruments,
            'notes_range': self.notes_range
        })

        cprint('Done Generating', 'green')

    def get_seed(self, nb_steps, step_length, number=1):
        """

        :param nb_steps:
        :param step_length:
        :param number:
        :return:
        """
        seeds = []
        with open(self.data_seed_pathlib / 'infos_dataset.p', 'rb') as dump_file:
            d = pickle.load(dump_file)
            all_shapes = d['all_shapes']
        for i in range(number):
            array_list = np.load(str(self.data_seed_pathlib / 'npy' / '{0}.npy'.format(
                np.random.randint(0, len(all_shapes))
            )), allow_pickle=True).item()['list']
            array = array_list[np.random.randint(0, len(array_list))]  # The song
            start = np.random.randint(0, (math.floor(array.shape[0] / step_length) - nb_steps)) * step_length
            seed = array[
                   start: start + nb_steps * step_length]  # (nb_steps * step_length, nb_intruments, input_size, 2)
            seed = np.reshape(seed, (nb_steps, step_length, seed.shape[1], seed.shape[2],
                                     seed.shape[3]))  # (nb_steps, step_length, nb_instruments, input_size, 2)
            seed = np.transpose(seed, (2, 0, 1, 3, 4))  # (nb_instruments, nb_steps, step_lenght, input_size, 2)
            seeds.append(seed)
        return seeds

    def compare_generation(self, max_length=None, no_duration=False, verbose=1):
        """

        :return:
        """
        # -------------------- Find informations --------------------
        if self.data_transformed_pathlib is None:
            self.data_transformed_pathlib = self.data_seed_pathlib
        nb_steps = int(self.model_id.split(',')[2])
        if self.my_sequence is None:
            self.my_sequence = MySequenceBeat(
                path=str(self.data_transformed_pathlib),
                nb_steps=nb_steps,
                batch_size=1,
                work_on=self.work_on)
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
            sample = np.concatenate((generated[:, :, l:, :, :, :], np.array(ms_input)),
                                    axis=1)  # (nb_instruments, 2, nb_steps, step_size, input_size, 2)

            # Generation
            preds = self.my_nn.generate(input=list(sample))

            # Reshape
            preds = np.asarray(preds).astype('float64')  # (nb_instruments, 2, length, 88, 2)
            preds_truth = np.array(ms_output)[:, :, np.newaxis, :, :,
                          :]  # (nb_instruments, 1, 1, step_size, input_size, 2)
            # if only one instrument
            if len(preds.shape) == 4:  # Only one instrument : output of nn not a list
                preds = preds[np.newaxis, :, :, :, :]
            if len(preds_truth.shape) == 4:  # Only one instrument : output of nn not a list
                preds_truth = preds_truth[np.newaxis, :, :, :, :]
            preds = midi_create.normalize_activation(preds)  # Normalize the activation part
            preds_helped = preds[:, 1, :, :, :][:, np.newaxis, np.newaxis, :, :,
                           :]  # (nb_instruments, 1, 1, lenght, 88, 2)
            preds = preds[:, 0, :, :, :][:, np.newaxis, np.newaxis, :, :, :]

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
                                                       notes_range=self.notes_range, no_duration=no_duration)
        # Helped
        generated_midi_final_helped = np.reshape(generated_helped, (
            generated_helped.shape[0], generated_helped.shape[2] * generated_helped.shape[3], generated_helped.shape[4],
            generated_helped.shape[5]))  # (nb_instruments, nb_step * length, 88 , 2)
        generated_midi_final_helped = np.transpose(generated_midi_final_helped,
                                            (0, 2, 1, 3))  # (nb_instruments, 88, nb_steps * length, 2)
        output_notes_list_helped = midi_create.matrix_to_midi(generated_midi_final_helped, instruments=self.instruments,
                                                       notes_range=self.notes_range, no_duration=no_duration)
        # Truth
        generated_midi_final_truth = np.reshape(generated_truth, (
            generated_truth.shape[0], generated_truth.shape[2] * generated_truth.shape[3], generated_truth.shape[4],
            generated_truth.shape[5]))  # (nb_instruments, nb_step * length, 88 , 2)
        generated_midi_final_truth = np.transpose(generated_midi_final_truth,
                                                   (0, 2, 1, 3))  # (nb_instruments, 88, nb_steps * length, 2)
        output_notes_list_truth = midi_create.matrix_to_midi(generated_midi_final_truth, instruments=self.instruments,
                                                       notes_range=self.notes_range, no_duration=no_duration)

        # ---------- find the name for the midi_file ----------
        self.get_new_save_midis_path()
        self.save_midis_pathlib.mkdir(parents=True, exist_ok=True)

        # -------------------- Save Results --------------------
        # Generated
        path_to_save = str(self.save_midis_pathlib / 'generated.mid')
        path_to_save_img = str(self.save_midis_pathlib / 'generated.jpg')
        if self.work_on == 'note':
            step_length = 1
        elif self.work_on == 'beat':
            step_length = g.step_per_beat
        elif self.work_on == 'measure':
            step_length = 4 * g.step_per_beat
        midi_create.print_informations(nb_steps=nb_steps * step_length, matrix=generated_midi_final,
                                       notes_list=output_notes_list, verbose=verbose)

        # Saving the midi file
        midi_create.save_midi(output_notes_list=output_notes_list, instruments=self.instruments,
                              path=path_to_save, )
        pianoroll.save_pianoroll(array=generated_midi_final,
                                 path=path_to_save_img,
                                 seed_length=nb_steps * step_length,
                                 instruments=self.instruments)

        # Helped
        path_to_save = str(self.save_midis_pathlib / 'generated_helped.mid')
        path_to_save_img = str(self.save_midis_pathlib / 'generated_helped.jpg')
        if self.work_on == 'note':
            step_length = 1
        elif self.work_on == 'beat':
            step_length = g.step_per_beat
        elif self.work_on == 'measure':
            step_length = 4 * g.step_per_beat
        midi_create.print_informations(nb_steps=nb_steps * step_length, matrix=generated_midi_final_helped,
                                       notes_list=output_notes_list_helped, verbose=verbose)

        # Saving the midi file
        midi_create.save_midi(output_notes_list=output_notes_list, instruments=self.instruments,
                              path=path_to_save, )
        pianoroll.save_pianoroll(array=generated_midi_final_helped,
                                 path=path_to_save_img,
                                 seed_length=nb_steps * step_length,
                                 instruments=self.instruments)

        # Truth
        path_to_save = str(self.save_midis_pathlib / 'generated_truth.mid')
        path_to_save_img = str(self.save_midis_pathlib / 'generated_truth.jpg')
        if self.work_on == 'note':
            step_length = 1
        elif self.work_on == 'beat':
            step_length = g.step_per_beat
        elif self.work_on == 'measure':
            step_length = 4 * g.step_per_beat
        midi_create.print_informations(nb_steps=nb_steps * step_length, matrix=generated_midi_final_truth,
                                       notes_list=output_notes_list_truth, verbose=verbose)

        # Saving the midi file
        midi_create.save_midi(output_notes_list=output_notes_list, instruments=self.instruments,
                              path=path_to_save, )
        pianoroll.save_pianoroll(array=generated_midi_final_truth,
                                 path=path_to_save_img,
                                 seed_length=nb_steps * step_length,
                                 instruments=self.instruments)

