from epicpath import EPath
import pickle
from termcolor import colored, cprint

from .MGInit import MGInit
from src.NN import Sequences
from src.NN import Models


class MGData(MGInit):
    
    def load_data(self, data_transformed_path=None, verbose=1):
        """

        :return: load the data
        """
        self.data_transformed_path = EPath(
            data_transformed_path) if data_transformed_path is not None else self.data_transformed_path
        self.input_param = {}
        with open(str(self.data_transformed_path / 'infos_dataset.p'), 'rb') as dump_file:
            d = pickle.load(dump_file)
            self.input_param['input_size'] = d['input_size']
            self.input_param['nb_instruments'] = d['nb_instruments']
            self.instruments = d['instruments']
            self.notes_range = d['notes_range']
            self.mono = d['mono']
            self.transposed = d['transposed']
        if verbose == 1:
            print('data at', colored(data_transformed_path, 'grey', 'on_white'), 'loaded')

    def change_batch_size(self, batch_size):
        if self.sequence is not None and self.batch != batch_size:
            self.batch = batch_size
            self.sequence.change_batch_size(batch_size=batch_size)

    def get_sequence(self, sequence_name=None, **kwargs):
        """

        :param sequence_name:
        :param kwargs:
        :return:
        """

        if sequence_name is not None:
            self.sequence = Sequences.from_name[sequence_name](**kwargs)
        else:
            self.sequence = Models.sequences[self.model_name](**kwargs)


