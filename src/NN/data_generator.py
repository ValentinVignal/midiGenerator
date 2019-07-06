import numpy as np
import tensorflow as tf
from pathlib import Path
import pickle
import functools


class MySequence(tf.keras.utils.Sequence):
    """

    """

    def __init__(self, path, nb_step, batch_size=5):
        """

        :param nb_files: nb files that can be read
        :param npy_path: the path to the folder of the npy files
        :param batch_size: batch size for training
        """
        self.path = path
        self.pathlib = Path(path)
        self.batch_size = batch_size  # bacth size
        self.npy_pathlib = self.pathlib / 'npy'
        self.npy_path = self.npy_pathlib.as_posix()  # Path for the npy folder
        self.nb_step = nb_step
        with open(self.pathlib / 'infos_dataset.p', 'rb') as dump_file:
            d = pickle.load(dump_file)
            self.nb_files = d['nb_files']  # nb available files in the dataset
            self.all_shapes = d['all_shapes']

        self.i_loaded = None  # number of the .npy already loaded
        self.npy_loaded = None  # npy file already loaded
        self.nb_file_per_npy = 100

        self.nb_elements = self.return_nb_elements(self.all_shapes)        # nb element available in the generator
        self.nb_elements = int(self.nb_elements / batch_size)
        self.all_len = self.know_all_len()
        print('MySequence instance initiated on the data {0}'.format(self.path))

    def __len__(self):
        return self.nb_elements

    def __getitem__(self, item):
        i_start = item * self.batch_size
        i, j, k = self.return_IJK(i_start)
        x = []
        y = []
        for s in range(self.batch_size):
            if i != self.i_loaded:
                self.i_loaded = i
                self.npy_loaded = np.load(str(self.npy_pathlib / '{0}.npy'.format(i)), allow_pickle=True).item()['list']
            x.append(self.npy_loaded[j][k: k + self.nb_step])
            y.append(self.npy_loaded[j][k+self.nb_step])
            k += 1
            if k == self.all_len[i][j]:
                k = 0
                j += 1
                if j == len(self.all_len[i]):
                    j = 0
                    i += 1
        x, y = np.asarray(x), np.asarray(y)

        return x, y

    def return_IJK(self, i_start):
        """

        :param i_start:
        :return: file i.npy song number j and step k to start
        """
        i = 0
        j = 0
        c = 0
        flag = True
        while flag:
            if c + self.all_len[i][j] > i_start:
                flag = False
            else:
                c += self.all_len[i][j]
                j += 1
                if j == len(self.all_len[i]):
                    j = 0
                    i += 1
        k = i_start - c
        return i, j, k

    def return_nb_elements(self, l, acc=0):
        """

        :param l:
        :param acc:
        :return:
        """

        if type(l) is list:
            for l2 in l:
                acc += self.return_nb_elements(l2, acc)
            return acc
        else:
            return acc + (l[0] - self.nb_step + 1)

    def know_all_len(self):
        def f_map(l):
            return functools.reduce(lambda x, y: x + y.shape[0], l)

        all_len = list(map(f_map, self.all_shapes))

        return all_len


