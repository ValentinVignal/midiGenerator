import numpy as np
import tensorflow as tf
from pathlib import Path
import pickle
import functools
from termcolor import colored
import math

import matplotlib.pyplot as plt

import src.global_variables as g
import src.image.pianoroll as pianoroll


class MySequence(tf.keras.utils.Sequence):
    """

    """

    def __init__(self, path, nb_steps, batch_size=4, work_on=g.work_on):
        """

        :param path: the path to the datas
        :param nb_steps: nb of steps in one frame
        :param batch_size: batch size for training
        :param work_on: can be 'beat' or 'measure'
        """
        self.pathlib = Path(path)
        self.batch_size = batch_size  # batch size
        self.npy_pathlib = self.pathlib / 'npy'
        self.nb_steps = nb_steps
        self.work_on = work_on
        self.step_size = None
        if work_on == 'note':
            self.step_size = 1
        elif work_on == 'beat':
            self.step_size = g.step_per_beat
        elif work_on == 'measure':
            self.step_size = 4 * g.step_per_beat

        with open(self.pathlib / 'infos_dataset.p', 'rb') as dump_file:
            d = pickle.load(dump_file)
            self.nb_files = d['nb_files']  # nb available files in the dataset
            self.all_shapes = d['all_shapes']  # [nb_files, nb_song_in_file], (length, nb_instrument, input_size, 2)

        self.i_loaded = None  # number of the .npy already loaded
        self.npy_loaded = None  # npy file already loaded
        self.nb_file_per_npy = g.nb_file_per_npy

        self.nb_elements = MySequence.return_nb_elements(self.all_shapes, self.step_size,
                                                         self.nb_steps)  # nb element available in the generator
        self.nb_elements = int(self.nb_elements / self.batch_size)
        self.all_len = self.know_all_len()

        self.noise = 0

        print('MySequence instance initiated on the data', colored(self.pathlib.as_posix(), 'grey', 'on_white'))

    # ---------- For Keras ----------

    def __len__(self):
        return self.nb_elements

    def __getitem__(self, item):
        i_start = item * self.batch_size
        i, j, k = self.return_ijk(i_start)
        x = []  # (batch, nb_steps * step_size, nb_instruments, input_size, 2)
        y = []  # (batch, nb_instruments, input_size, 2)
        for s in range(self.batch_size):
            if i != self.i_loaded:
                self.i_loaded = i
                self.npy_loaded = np.load(str(self.npy_pathlib / f'{i}.npy'), allow_pickle=True).item()['list']
            x.append(
                self.npy_loaded[j][k: k + (self.nb_steps * self.step_size)]
            )  # (batch, nb_steps * step_size, nb_instruments, input_size, 2)
            y.append(
                self.npy_loaded[j][k + (self.nb_steps * self.step_size): k + (
                        self.nb_steps * self.step_size) + self.step_size]
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

        x = np.transpose(x, (3, 0, 1, 2, 4, 5))  # (nb_instruments, batch, nb_steps, step_size, input_size, 2)
        y = np.transpose(y, (2, 0, 1, 3, 4))  # (nb_instruments, batch, step_size, input_size, 2)

        if self.noise is not None and self.noise > 0:
            # Creation of the noise
            noise = np.random.binomial(n=1, p=self.noise, size=x[..., 0].shape)
            x[..., 0] = np.abs(x[..., 0] - noise)

        return list(x), list(y)

    # ---------- For the user ----------

    def set_noise(self, noise):
        self.noise = noise

    def change_batch_size(self, batch_size):
        if self.batch_size != batch_size:
            self.batch_size = batch_size

            self.nb_elements = MySequence.return_nb_elements(self.all_shapes, self.step_size,
                                                             self.nb_steps)  # nb element available in the generator
            self.nb_elements = int(self.nb_elements / self.batch_size)
            self.all_len = self.know_all_len()

    # --------- Helper functions ---------

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
    def return_nb_elements(l, step_size, nb_steps):
        """

        :param l:
        :param step_size:
        :param nb_steps:
        :return: The number of useful element in the dataset
        """
        if type(l) is list:
            acc = 0
            for l2 in l:
                acc += MySequence.return_nb_elements(l2, step_size, nb_steps)
            return acc
        else:
            return int(l[0] / step_size) - nb_steps  # not - nb_steps + 1 because of the y (true tab)

    def know_all_len(self):
        """

        :return: all the length of all files
        """

        def f_map(l):
            return functools.reduce(lambda x, y: x + int(y[0] / self.step_size) - self.nb_steps,
                                    l,
                                    0)  # not -self.nb_steps + 1 because of the y (true tab)

        all_len = list(map(f_map, self.all_shapes))

        return all_len


# ------------------------------------------------------------


class MySequenceReduced(tf.keras.utils.Sequence):
    """
    This class is use to train with a validation split
    """
    def __init__(self, my_sequence, indexes):
        self.my_sequence = my_sequence
        self.indexes = indexes

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, index):
        return self.my_sequence[self.indexes[index]]


def get_train_valid_mysequence(my_sequence, validation_split=0.0):
    """

    :param my_sequence:
    :param validation_split:
    :return:
    """
    all_len = len(my_sequence)
    len_valid = math.ceil(all_len * validation_split)
    permutation = np.random.permutation(np.arange(all_len))
    indexes_valid = permutation[:len_valid]
    indexes_train = permutation[len_valid:]
    my_sequence_valid = MySequenceReduced(my_sequence, indexes_valid)
    my_sequence_train = MySequenceReduced(my_sequence, indexes_train)
    return my_sequence_train, my_sequence_valid


# ------------------------------------------------------------

class SeeMySequence:

    def __init__(self, path, nb_steps, work_on):
        self.my_sequence = MySequence(path=path, nb_steps=nb_steps, batch_size=1, work_on=work_on)
        # my_sequence[i] -> tuple [0] = x, [1] = y
        #                   -> [ list ] (nb_instruments)
        #                       -> np array (batch, nb_steps, step_size, input_size, 2)
        self.nb_instruments = np.array(self.my_sequence[0][0]).shape[0]
        self.nb_steps = nb_steps
        self.input_size = np.array(self.my_sequence[0][0]).shape[4]

        self.colors = None
        self.new_colors()

    def set_noise(self, noise):
        self.my_sequence.set_noise(noise)

    def new_colors(self):
        # Colors
        self.colors = pianoroll.return_colors(self.nb_instruments)

    def show(self, indice, nb_rows=3, nb_colums=4):
        nb_images = nb_rows * nb_colums
        fig = plt.figure()
        for ind in range(nb_images):
            x, y = self.my_sequence[indice + ind]
            # activations
            x = np.array(x)[:, 0, :, :, :, 0]  # (nb_instruments, nb_steps, step_size, input_size)
            x = np.reshape(x,
                           (x.shape[0], x.shape[1] * x.shape[2], x.shape[3])
                           )  # (nb_instruments, nb_steps * step_size, input_size)
            y = np.array(y)[:, 0, :, :, 0]  # (nb_instruments, step_size, input_size)
            np.place(x, 0.5 <= x, 1)
            np.place(x, x < 0.5, 0)
            np.place(y, 0.5 <= y, 1)
            np.place(y, y < 0.5, 0)

            all = np.zeros((x.shape[1] + y.shape[1], self.input_size, 3))
            all[- y.shape[1]:] = 50

            for inst in range(self.nb_instruments):
                for j in range(self.input_size):
                    for i in range(x.shape[1]):
                        if x[inst, i, j] == 1:
                            all[i, j] = self.colors[inst]
                    for i in range(y.shape[1]):
                        if y[inst, i, j] == 1:
                            all[x.shape[1] + i, j] = self.colors[inst]
            all = (np.flip(np.transpose(all, (1, 0, 2)), axis=0)).astype(np.int)

            fig.add_subplot(nb_rows, nb_colums, ind + 1)
            plt.imshow(all)
        plt.show()

    def __len__(self):
        return len(self.my_sequence)

    def __getitem__(self, item):
        return self.my_sequence[item]


