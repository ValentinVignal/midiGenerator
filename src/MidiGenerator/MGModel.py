from pathlib import Path
import pickle
from termcolor import cprint, colored
import shutil

from src.NN.KerasNeuralNetwork import KerasNeuralNetwork
from src import GlobalVariables as g
import src.text.summary as summary
from .MGInit import MGInit


class MGModel(MGInit):

    def new_nn_model(self, model_id, work_on=None, opt_param=None, model_options=None, loss_options=None,
                     print_model=True, predict_offset=None):
        """

        :param loss_options:
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
            self.work_on = g.mg.work_on if self.work_on is None else self.work_on
        else:
            self.work_on = work_on

        step_length = g.mg.work_on2nb(self.work_on)
        self.get_new_full_name()
        self.predict_offset = g.train.predict_offset if predict_offset is None else predict_offset

        opt_param = {'lr': g.nn.lr, 'name': 'adam'} if opt_param is None else opt_param

        self.keras_nn = KerasNeuralNetwork()
        self.keras_nn.new_model(
            model_id=self.model_id,
            step_length=step_length,
            input_param=self.input_param,
            opt_param=opt_param,
            model_options=model_options,
            loss_options=loss_options
        )
        if print_model:
            self.print_model()

    def recreate_model(self, id, with_weights=True, print_model=True):
        """
        create a new model witht the same options as the saved model and then load the weights (if with_weights==True)
        :param id:
        :param with_weigths: if we have to load the weight of the model
        :param print_model:
        :return:
        """
        name, model_id, work_on_letter, total_epochs, indice = id.split('-')
        path_to_load = Path('saved_models',
                            f'{name}-m({model_id})-wo({work_on_letter})-e({total_epochs})-({indice})')
        with open(str(path_to_load / 'infos.p'), 'rb') as dump_file:
            d = pickle.load(dump_file)
            # Model
            self.model_id = d['model_id']
            self.work_on = d['work_on']
            self.input_param = d['input_param']
            # Data
            self.instruments = d['instruments']
            self.notes_range = d['notes_range']
            self.mono = d['mono']
            self.data_transformed_path = d['data_transformed_path']
            # Logistic
            self.total_epochs = d['epochs'] if with_weights else 0
            self.full_name = d['full_name'] if with_weights else self.get_full_name(indice)

        self.keras_nn = KerasNeuralNetwork()
        self.keras_nn.recreate((path_to_load / 'MyNN').as_posix(), with_weights=with_weights)

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
                            f'{self.name}-m({self.model_id})-wo({work_on_letter})-e({self.total_epochs})-({indice})')
        self.keras_nn.load_weights(str(path_to_load / 'MyNN'))
        self.print_model()
        print('Weights of the', colored(f'{id}', 'white', 'on_blue'), 'model loaded')

    # --------------------------------------------------

    def print_model(self):
        print(self.keras_nn.model.summary())

    def save_model(self, path=None):
        """

        :param path: path were to save the model, if None it will be at self.saved_model_path
        :return:
        """
        # Where to save the model
        path_to_save = self.saved_model_path if path is None else Path(path)
        # Clean the folder if already in use
        shutil.rmtree(path_to_save.as_posix(), ignore_errors=True)     # Delete it if it exists
        path_to_save.mkdir(parents=True, exist_ok=True)  # Creation of this folder
        self.keras_nn.save(str(path_to_save / 'MyNN'))
        with open(str(path_to_save / 'infos.p'), 'wb') as dump_file:
            pickle.dump(
                dict(
                    # For new_nn_model
                    model_id=self.model_id,
                    work_on=self.work_on,
                    input_param=self.input_param,
                    # data
                    instruments=self.instruments,
                    notes_range=self.notes_range,
                    mono=self.mono,
                    data_transformed_path=self.data_transformed_path,
                    # logistic
                    epochs=self.total_epochs,
                    full_name=self.full_name
                ), dump_file)
        self.keras_nn.save_tensorboard_plots(path_to_save / 'plots')

        summary.summarize(
            # Function params
            path=path_to_save,
            title=self.full_name,
            # Summary params
            epochs=self.total_epochs,
            input_param=self.input_param,
            instruments=self.instruments,
            notes_range=self.notes_range,
            work_on=self.work_on,
            mono=self.mono
        )

        print(colored(f'Model saved in {path_to_save}', 'green'))
        return path_to_save

    # --------------------------------------------------

    def print_weights(self):
        """
        Print the weights
        :return:
        """
        for layer in self.keras_nn.model.layers:
            weights = layer.get_weights()  # list of numpy arrays
            print('Weights:', weights)

