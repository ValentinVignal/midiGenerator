import tensorflow as tf
from pathlib import Path
import pickle
import functools
import numpy as np

from src import GlobalVariables as g


class KerasSequence(tf.keras.utils.Sequence):
    """

    """

    def __init__(self, path, nb_steps, batch_size=4, work_on=g.mg.work_on, noise=0, replicate=False,
                 predict_offset=g.train.predict_offset):
        """

        :type predict_offset:
        :param path: The path to the data
        :param nb_steps: The number of steps in the inputs
        :param work_on: if it is on note/beat/measure
        :param replicate: if true, input = output, else output = last step of input + 1
        """
        # -------------------- Attribut --------------------
        self.predict_offset = 0 if replicate else predict_offset
        self.path = Path(path)
        self.npy_path = self.path / 'npy'
        self.nb_steps = nb_steps
        self.work_on = work_on
        self.step_size = g.mg.work_on2nb(work_on)
        self.noise = noise
        self.batch_size = batch_size
        self.replicate = replicate

        # -------------------- Technical Attributs --------------------

        with open(self.path / 'infos_dataset.p', 'rb') as dump_file:
            d = pickle.load(dump_file)
            self.nb_files = d['nb_files']  # nb available files in the dataset
            self.all_shapes = d['all_shapes']  # [nb_files, nb_song_in_file], (length, nb_instrument, input_size, 2)
            self.nb_files_per_npy = d['nb_files_per_npy']
            self.nb_instruments = d['nb_instruments']

        self.i_loaded = None  # number of the .npy already loaded
        self.npy_loaded = None  # npy file already loaded
        self.nb_file_per_npy = g.midi.nb_files_per_npy

        self.nb_elements = KerasSequence.return_nb_elements(
            l=self.all_shapes,
            step_size=self.step_size,
            nb_steps=self.nb_steps,
            predict_offset=predict_offset
        )  # nb element available in the generator
        self.nb_elements = int(self.nb_elements / self.batch_size)
        self.all_len = self.know_all_len(
            all_shapes=self.all_shapes,
            step_size=self.step_size,
            nb_steps=self.nb_steps,
            predict_offset=self.predict_offset
        )

    # -------------------- For Keras --------------------

    def __len__(self):
        return self.nb_elements

    def __getitem__(self, item):
        i_start = item
        i, j, k = self.return_ijk(i_start)
        x = []  # (batch, nb_steps * step_size, nb_instruments, input_size, 2)
        y = []  # (batch, nb_instruments, input_size, 2)
        for s in range(self.batch_size):
            x_start, x_stop = k, k + self.nb_steps * self.step_size
            y_start = k + (self.nb_steps + self.predict_offset - 1) * self.step_size
            y_stop = y_start + self.step_size

            if i != self.i_loaded:
                self.i_loaded = i
                self.npy_loaded = np.load(str(self.npy_path / f'{i}.npy'), allow_pickle=True).item()['list']
            x.append(
                self.npy_loaded[j][x_start: x_stop]
            )  # (batch, nb_steps * step_size, nb_instruments, input_size, 2)
            y.append(
                self.npy_loaded[j][y_start: y_stop]
            )  # (batch, step_size, nb_instruments, input_size, 2)
            k += self.step_size
            if k >= self.all_shapes[i][j][0] - ((self.nb_steps + 1) * self.step_size):
                k = 0
                j += 1
                if j == len(self.all_shapes[i]):
                    j = 0
                    i += 1
        x, y = np.asarray(x), np.asarray(y)
        batch, nb_steps_size, nb_instruments, input_size, channels = x.shape
        x = np.reshape(x, (self.batch_size, self.nb_steps, self.step_size, nb_instruments, input_size, channels))
        # x = (batch, nb_steps, step_size, nb_instruments, input_size, channels)
        y = np.expand_dims(y, axis=1)
        # y = (batch, nb_steps=1, step_size, nb_instruments, input_size, channels)

        x = np.transpose(x, (3, 0, 1, 2, 4, 5))  # (nb_instruments, batch, nb_steps, step_size, input_size, 2)
        y = np.transpose(y, (3, 0, 1, 2, 4, 5))  # (nb_instruments, batch, nb_steps=1, step_size, input_size, 2)
        if self.replicate:
            y = np.copy(x)

        if self.noise is not None and self.noise > 0:
            # Creation of the noise
            noise = np.random.binomial(n=1, p=self.noise, size=x[..., 0].shape)
            x[..., 0] = np.abs(x[..., 0] - noise)

        return x, y

    # -------------------- For the user --------------------

    def set_noise(self, noise):
        self.noise = noise

    def change_batch_size(self, batch_size):
        if self.batch_size != batch_size:
            self.batch_size = batch_size

            self.nb_elements = KerasSequence.return_nb_elements(
                self.all_shapes, self.step_size,
                self.nb_steps,
                self.predict_offset
            )  # nb element available in the generator
            self.nb_elements = int(self.nb_elements / self.batch_size)
            self.all_len = self.know_all_len(
                all_shapes=self.all_shapes,
                step_size=self.step_size,
                nb_steps=self.nb_steps,
                predict_offset=self.predict_offset
            )

    # -------------------- Helper functions --------------------

    def return_ijk(self, i_start):
        """

        :param i_start:
        :return: file i.npy song number j and step k to start
        """
        i = 0
        j = 0
        c = 0
        flag = True
        while flag:
            if c + self.all_len[i] > i_start:
                flag = False
            else:
                c += self.all_len[i]
                i += 1
        flag = True
        while flag:
            if c + int(self.all_shapes[i][j][0] / self.step_size) - self.nb_steps > i_start:
                flag = False
            else:
                c += int(self.all_shapes[i][j][0] / self.step_size) - self.nb_steps
                j += 1
        k = (i_start - c) * self.step_size
        return i, j, k

    @staticmethod
    def return_nb_elements(l, step_size, nb_steps, predict_offset=g.train.predict_offset):
        """

        :param predict_offset:
        :param l:
        :param step_size:
        :param nb_steps:
        :return: The number of useful element in the dataset
        """
        if type(l) is list:
            acc = 0
            for l2 in l:
                acc += KerasSequence.return_nb_elements(l2, step_size, nb_steps, predict_offset)
            return acc
        else:
            return int(
                l[0] / step_size) - nb_steps + 1 - predict_offset  # not - nb_steps + 1 because of the y (true tab)

    @staticmethod
    def know_all_len(all_shapes, step_size, nb_steps, predict_offset=g.train.predict_offset):
        """

        :return: all the length of all files
        """

        def f_map(l):
            return functools.reduce(lambda x, y: x + int(y[0] / step_size) - nb_steps + 1 - predict_offset,
                                    l,
                                    0)  # not -self.nb_steps + 1 because of the y (true tab)

        all_len = list(map(f_map, all_shapes))

        return all_len

    def __del__(self, *args, **kwargs):
        del self.npy_loaded
