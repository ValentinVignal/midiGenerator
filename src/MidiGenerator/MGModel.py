from pathlib import Path
import pickle
from termcolor import cprint, colored

from src.NN.KerasNeuralNetwork import KerasNeuralNetwork
import src.global_variables as g
import src.text.summary as summary
from .MGInit import MGInit


class MGModel(MGInit):

    def new_nn_model(self, model_id, work_on=None, opt_param=None, model_options=None,
                     print_model=True):
        """

        :param model_id: modelName;modelParam;nbSteps
        :param work_on:
        :param opt_param:
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
                                model_options=model_options)
        if print_model:
            self.print_model()

    def recreate_model(self, id, with_weigths=True, print_model=True):
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
            self.data_transformed_path = d['data_transformed_path']

        self.keras_nn = KerasNeuralNetwork()
        self.keras_nn.recreate((path_to_load / 'MyNN').as_posix())

        if with_weigths:
            self.keras_nn.load_weights((path_to_load / 'MyNN').as_posix())
            self.total_epochs = int(total_epochs)
            self.get_full_name(indice)
        if print_model:
            self.print_model()

    def load_weights(self, id, keep_name=True):
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

    def save_model(self, path=None):
        """

        :param path: path were to save the model, if Nonem it will be at self.saved_model_path
        :return:
        """
        path_to_save = self.saved_model_path if path is None else Path(path)
        path_to_save.mkdir(parents=True, exist_ok=True)  # Creation of this folder
        self.saved_model_path.mkdir(parents=True, exist_ok=True)
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
                'data_transformed_path': self.data_transformed_path,
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

