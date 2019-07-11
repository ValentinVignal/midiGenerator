import numpy as np
import tensorflow as tf
from pathlib import Path
import pickle
import functools


class MySequence(tf.keras.utils.Sequence):
    """

    """

    def __init__(self, path, nb_steps, batch_size=5):
        """

        :param nb_files: nb files that can be read
        :param npy_path: the path to the folder of the npy files
        :param batch_size: batch size for training
        """
        self.path = path
        self.pathlib = Path(path)
        self.batch_size = batch_size  # batch size
        self.npy_pathlib = self.pathlib / 'npy'
        self.npy_path = self.npy_pathlib.as_posix()  # Path for the npy folder
        self.nb_steps = nb_steps
        with open(self.pathlib / 'infos_dataset.p', 'rb') as dump_file:
            d = pickle.load(dump_file)
            self.nb_files = d['nb_files']  # nb available files in the dataset
            self.all_shapes = d['all_shapes']

        self.i_loaded = None  # number of the .npy already loaded
        self.npy_loaded = None  # npy file already loaded
        self.nb_file_per_npy = 100

        self.nb_elements = self.return_nb_elements(self.all_shapes)        # nb element available in the generator
        self.nb_elements = int(self.nb_elements / self.batch_size)
        self.all_len = self.know_all_len()
        print('MySequence instance initiated on the data {0}'.format(self.path))

    def __len__(self):
        return self.nb_elements

    def __getitem__(self, item):
        i_start = item * self.batch_size
        i, j, k = self.return_IJK(i_start)
        x = []      # (batch, nb_steps, nb_instruments, input_size)
        y = []      # (batch, nb_instruments, input_size)
        for s in range(self.batch_size):
            if i != self.i_loaded:
                self.i_loaded = i
                self.npy_loaded = np.load(str(self.npy_pathlib / '{0}.npy'.format(i)), allow_pickle=True).item()['list']
            x.append(self.npy_loaded[j][k: k + self.nb_steps])
            y.append(self.npy_loaded[j][k + self.nb_steps])
            k += 1
            if k == self.all_shapes[i][j][0] - self.nb_steps + 1:
                k = 0
                j += 1
                if j == len(self.all_shapes[i]):
                    j = 0
                    i += 1
        x, y = np.asarray(x), np.asarray(y)

        x = np.transpose(x, (2, 0, 1, 3))       # (nb_instruments, batch, nb_steps, input_size)
        y = np.transpose(y, (1, 0, 2))      # (nb_instruments, batch, input_size)

        return list(x), list(y)

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
            if c + self.all_len[i] > i_start:
                flag = False
            else:
                c += self.all_len[i]
                i += 1
        flag = True
        while flag:
            if c + self.all_shapes[i][j][0] - self.nb_steps > i_start:
                flag = False
            else:
                c += self.all_shape[i][j][0] - self.nb_steps
                j += 1
        k = i_start - c
        return i, j, k

    def return_nb_elements(self, l):
        """

        :param l:
        :param acc:
        :return:
        """

        if type(l) is list:
            acc = 0
            for l2 in l:
                acc += self.return_nb_elements(l2)
            return acc
        else:
            print(l)
            return l[0] - self.nb_steps # not -self.nb_steps + 1 because of the y (true tab)

    def know_all_len(self):
        def f_map(l):
            return functools.reduce(lambda x, y: x + y[0] - self.nb_steps, l, 0)    # not -self.nb_steps + 1 because of the y (true tab)

        all_len = list(map(f_map, self.all_shapes))

        return all_len


