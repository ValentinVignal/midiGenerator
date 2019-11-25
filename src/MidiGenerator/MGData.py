from pathlib import Path
import pickle
from termcolor import colored, cprint

from .MGInit import MGInit


class MGData(MGInit):
    
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
