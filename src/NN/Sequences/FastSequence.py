""""
Faster sequence: save all the data in a temporary folder and then delete it
Only load the files and give the inputs, doesn't do any processing for a faster training
"""
import tensorflow as tf
from pathlib import Path
import shutil
import numpy as np
import math


class FastSequence(tf.keras.utils.Sequence):
    """

    """
    def __init__(self, sequence, nb_steps_per_file=100, batch_size=None, dataset_folder_path=None):
        """


        :param sequence: Sequence to copy (Keras Sequence)
        :param nb_steps_per_file:
        :param batch_size:

        Attributs:

        :var folder_path: path to the folder containing the npy files
        :var token_path: path to the .txt file which is used as a token
        :var batch_size: the batch size
        :var nb_steps_per_file: number of steps saved in a .npy file
        :var nb_steps: total number of steps in this dataset
        :var npy_loaded: number of the file already loaded
        :var x_loaded: inputs array already loaded
        :var y_loaded: outputs array already loaded
        :var has_mask = if the sequence is supposed to have a mask in the inputs
        :var mask_loaded = masks array already loaded


        """
        # Create the folder temp im the dataset_path
        if dataset_folder_path is None:
            # Find it from the given sequence
            data_temp_folder = sequence.path.parent / 'temp'
        else:
            data_temp_folder = Path(dataset_folder_path) / 'temp'
        data_temp_folder.mkdir(exist_ok=True, parents=True)
        self.folder_path, self.token_path = self.get_token_path_file(data_temp_folder)

        # Set up batch size number
        self.batch_size = sequence.batch_size if batch_size is None else batch_size

        # Set up nb_steps_per_file number
        old_batch_size = sequence.batch_size
        sequence.change_batch_size(1)
        self.nb_steps_per_file = len(sequence) if nb_steps_per_file is None else nb_steps_per_file
        self.nb_steps = len(sequence)
        sequence.change_batch_size(old_batch_size)

        # if a mask is given by the sequence
        self.has_mask = len(sequence[0][0]) == len(sequence[0][1]) + 1

        # ----------------------------------------
        # Create the replicated dataset
        self.replicate_dataset(sequence)

        # ----------------------------------------
        # Create the variable for loading
        self.npy_loaded = None
        self.x_loaded = None
        self.y_loaded = None
        self.mask_loaded = None

    @staticmethod
    def get_token_path_file(temp_folder):
        temp_folder.mkdir(exist_ok=True, parents=True)
        i = 0
        name = 'fast_sequence_{0}'
        token_name = 'token_fast_sequence_{0}.txt'
        # Find the name
        while (temp_folder / name.format(i)).exists() or (temp_folder / token_name.format(i)).exists():
            i += 1
        # Create token and folder as soon as found
        with open(temp_folder / token_name.format(i), 'w') as f:
            f.write(f'token file for sequence')
        (temp_folder / name.format(i)).mkdir(exist_ok=True, parents=True)
        # Return the paths
        return temp_folder / name.format(i), temp_folder / token_name.format(i)

    def replicate_dataset(self, sequence):
        print('Replicate dataset for FastSequence instance')
        old_batch_size = sequence.batch_size
        sequence.change_batch_size(1)
        nb_inputs, nb_outputs = len(sequence[0][0]), len(sequence[0][1])
        for file_number in range(int(math.ceil(len(sequence) / self.nb_steps_per_file))):
            # Load all the data
            array_list_x = [[] for _ in range(sequence.nb_instruments)]
            array_list_y = [[] for _ in range(sequence.nb_instruments)]
            if self.has_mask:
                array_list_mask = []
            for i in range(self.nb_steps_per_file):
                sequence_index = self.nb_steps_per_file * file_number + i
                if sequence_index >= len(sequence):
                    break
                list_x, list_y = sequence[sequence_index]
                for inst in range(sequence.nb_instruments):
                    array_list_x[inst].append(list_x[inst][0])     # List(nb_instruments, batch)[(midi_shape)]
                    array_list_y[inst].append(list_y[inst][0])     # List(nb_instruments, batch)[(midi_shape)]
                if self.has_mask:
                    array_list_mask.append(list_x[-1][0])
            x = np.asarray(array_list_x)        # (nb_instruments, batch, midi_shape)
            y = np.asarray(array_list_y)        # (nb_instruments, batch, midi_shape)
            if self.has_mask:
                mask = np.array(array_list_mask)
            # Save it in a npy file
            np.save(self.folder_path / f'x_{file_number}.npy', x)
            np.save(self.folder_path / f'y_{file_number}.npy', y)
            if self.has_mask:
                np.save(self.folder_path / f'mask_{file_number}.npy', mask)
        sequence.change_batch_size(old_batch_size)

    def __len__(self):
        return self.nb_steps // self.batch_size

    def __del__(self):
        del self.x_loaded
        del self.y_loaded
        del self.mask_loaded
        # Delete the token file
        if self.token_path.exists():
            self.token_path.unlink()
        # Delete the replicated dataset
        shutil.rmtree(self.folder_path, ignore_errors=True)

    def __getitem__(self, item):
        np_index = self.batch_size * item
        # If we don't have the correct file, we load if
        print('item', item, 'np_index', np_index, 'npy loaded', self.npy_loaded)
        if np_index // self.nb_steps_per_file != self.npy_loaded:
            self.load(np_index // self.nb_steps_per_file)
        # Check if with need to load 2 files
        need_2_files = np_index // self.nb_steps_per_file != (np_index + self.batch_size - 1) // self.nb_steps_per_file
        file_index = np_index % self.nb_steps_per_file
        x, y, mask = self.get_xy(file_index)
        if need_2_files:
            nb_missing = self.batch_size - (self.nb_steps_per_file - file_index - 1)
            self.load(self.npy_loaded + 1)
            x2, y2, mask2 = self.get_xy(0, nb_missing)
            x = np.concatenate([x, x2], axis=1)
            y = np.concatenate([y, y2], axis=1)
            if self.has_mask:
                mask = np.concatenate([mask, mask2], axis=0)
        if self.has_mask:
            return list(x) + [mask], list(y)
        else:
            return list(x), list(y)

    def load(self, i):
        self.npy_loaded = i
        self.x_loaded = np.load(self.folder_path / f'x_{self.npy_loaded}.npy')
        self.y_loaded = np.load(self.folder_path / f'y_{self.npy_loaded}.npy')
        if self.has_mask:
            self.mask_loaded = np.load(self.folder_path / f'mask_{self.npy_loaded}.npy')

    def get_xy(self, index, nb_el=None):
        nb_el = self.batch_size if nb_el is None else nb_el
        x = self.x_loaded[:, index: index + nb_el]
        y = self.y_loaded[:, index: index + nb_el]
        mask = self.mask_loaded[index: index + nb_el] if self.has_mask else None
        return x, y, mask


















