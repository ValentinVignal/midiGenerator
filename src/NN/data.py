import numpy as np
import tensorflow as tf
from pathlib import Path


class MySequence(tf.keras.utils.Sequence):
    """

    """

    def __init__(self, nb_files, npy_path, nb_step, batch_size=5):
        """

        :param nb_files: nb files that can be read
        :param npy_path: the path to the folder of the npy files
        :param batch_size: batch size for training
        """
        self.nb_files = nb_files  # nb available files in the dataset
        self.batch_size = batch_size  # bacth size
        self.npy_path = npy_path  # Path for the npy folder
        self.npy_pathlib = Path(npy_path)
        self.nb_step = nb_step

        self.i_loaded = None  # number of the .npy already loaded
        self.npy_loaded = None  # npy file already loaded
        self.nb_file_per_npy = 100

        self.nb_elements, self.all_len = self.know_size(npy_path)        # nb element available in the generator
        self.nb_elements = int(self.nb_elements / batch_size)

    def __len__(self):
        return self.nb_files

    def __getitem__(self, item):
        i_start = item * self.batch_size
        i, j, k = self.return_IJK(i_start)
        x = []
        y = []
        for s in range(self.batch_size):
            if i != self.i_loaded:
                self.i_loaded = i
                self.npy_loaded = np.load(str(self.npy_pathlib / '{0}.npy'.format(i))).item()
            x.append(self.npy_loaded[j][k: k + self.nb_step])
            y.append(self.npy_loaded[j][k+self.nb_step])
            k += 1
            if k == self.all_len[i][j]:
                k = 0
                j += 1
                if j == len(self.all_len[i]):
                    j = 0
                    i += 1

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


    def know_size(self, npy_path):
        """

        :param npy_path: the path where the .npy files are
        :return: the number of elements available in the generator and the the length of all files (speaking in
        number of elements)
        """
        npy_pathlib = Path(npy_path)
        nb_seq = 0
        all_len = []
        i = 0
        npy = None
        file = str(npy_pathlib / '{0}.npy'.format(i))
        while file.exists():
            npy = np.load(file).item()
            len_f = []
            for j in range(len(npy)):
                l = int((len(npy[j]) - 1) / self.nb_step)
                nb_seq += l
                len_f.append(l)
            all_len.append(len_f)
            i += 1
            file = npy_pathlib / '{0}.npy'.format(i)
        self.i_loaded = i - 1
        self.npy_loaded = npy
        return nb_seq, all_len
