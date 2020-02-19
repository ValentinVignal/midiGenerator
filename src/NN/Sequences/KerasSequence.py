import tensorflow as tf
from epicpath import EPath
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
        self.predict_offset = 0 if replicate else predict_offset  # The next step as y
        self.path = EPath(path)  # Path to the dataset
        self.npy_path = self.path / 'npy'  # npy folder path
        self.nb_steps = nb_steps  # Number of steps in the x
        self.work_on = work_on  # measure/beat/note
        self.step_size = g.mg.work_on2nb(work_on)  # The size of a step
        self.noise = noise
        self.batch_size = batch_size
        self.replicate = replicate  # Boolean, if True, y = x

        self.nb_songs = None  # Number of songs in the dataset
        self.all_shapes = None  # all the size of the npy array
        # List(nb_files, nb_song_in_file)[(length, nb_instruments, input_size, channels)
        self.nb_songs_per_npy_file = None  # Maximum number of songs in a .npy file
        self.nb_instruments = None  # Number of instruments
        self.file_loaded = None  # The .npy file currently loaded in memory
        self.npy_loaded = None  # The value of the .npy file currently loaded in memory
        self._nb_elements_available = None  # number of steps in the all dataset
        self._nb_elements_available_per_file = None  # All the len of the songs List(nb_files)[int]
        self._nb_elements_available_per_song = None  # All the len of the songs List(nb_files, nb_songs_per_file)[int]

        # -------------------- Technical Attributs --------------------

        with open(self.path / 'infos_dataset.p', 'rb') as dump_file:
            d = pickle.load(dump_file)
            self.nb_songs = d['nb_files']  # nb available files in the dataset
            self.all_shapes = d['all_shapes']  # [nb_files, nb_song_in_file], (length, nb_instrument, input_size, 2)
            self.nb_songs_per_npy_file = d['nb_files_per_npy']
            self.nb_instruments = d['nb_instruments']

        self.file_loaded = None  # number of the .npy already loaded
        self.npy_loaded = None  # npy file already loaded

    @classmethod
    def getter(cls, **params):
        """

        :param params:
        :return:
        """

        def getter(*args, **kwargs):
            return cls(*args, **params, **kwargs)

        return getter

    def get_init_params(self):
        """

        :return:
        """
        return dict(
            path=self.path,
            nb_steps=self.nb_steps,
            batch_size=self.batch_size,
            work_on=self.work_on,
            noise=self.noise,
            replicate=self.replicate,
            predict_offset=self.predict_offset
        )

    @property
    def len(self):
        return self.nb_elements_available // self.batch_size

    @property
    def nb_elements_available(self):
        if self._nb_elements_available is None:
            self.nb_elements_available = self.return_nb_elements_available()
        return self._nb_elements_available

    @nb_elements_available.setter
    def nb_elements_available(self, nb_elements_available):
        self._nb_elements_available = nb_elements_available

    @property
    def nb_elements_available_per_file(self):
        if self._nb_elements_available_per_file is None:
            self.nb_elements_available_per_file = self.get_nb_elements_available_per_file()
        return self._nb_elements_available_per_file

    @nb_elements_available_per_file.setter
    def nb_elements_available_per_file(self, nb_elements_available_per_file):
        self._nb_elements_available_per_file = nb_elements_available_per_file

    @property
    def nb_elements_available_per_song(self):
        if self._nb_elements_available_per_song is None:
            self.nb_elements_available_per_song = self.get_nb_elements_available_per_song()
        return self._nb_elements_available_per_song

    @nb_elements_available_per_song.setter
    def nb_elements_available_per_song(self, nb_elements_available_per_song):
        self._nb_elements_available_per_song = nb_elements_available_per_song

    # -------------------- For Keras --------------------

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        return self.get_item(item)

    def get_item(self, item):
        i_start = item * self.batch_size
        file, song, index = self.return_ijk(i_start)
        # file = the number of the .npy file to open
        # song = the number of the song
        # index = the index in the song
        x = []  # (batch, nb_steps * step_size, nb_instruments, input_size, 2)
        y = []  # (batch, nb_instruments, input_size, 2)
        for s in range(self.batch_size):
            x_start, x_stop = index * self.step_size, (index + self.nb_steps) * self.step_size
            y_start = (index + self.nb_steps + self.predict_offset - 1) * self.step_size
            y_stop = y_start + self.step_size

            # If we the file needed is not the one already loaded, then we should load it
            if file != self.file_loaded:
                self.file_loaded = file
                self.npy_loaded = np.load(str(self.npy_path / f'{file}.npy'), allow_pickle=True).item()['list']
            x.append(
                self.npy_loaded[song][x_start: x_stop]
            )  # (batch, nb_steps * step_size, nb_instruments, input_size, 2)
            y.append(
                self.npy_loaded[song][y_start: y_stop]
            )  # (batch, step_size, nb_instruments, input_size, 2)
            # For the next step of the batch, we increment the index
            index += 1
            # If this is the end of the song, we take the next one
            if index >= self.nb_elements_available_per_song[file][song]:
                index = 0
                song += 1
                # if this is the end of the file, we take the next one
                if song == len(self.nb_elements_available_per_song[file]):
                    song = 0
                    file += 1
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

    # -------------------- Helper functions --------------------

    def return_ijk(self, i_start):
        """

        :param i_start:
        :return: file i.npy song number j and step k to start
        """
        file = 0  # Number of the file
        song = 0  # number of the song in the file
        index = 0  # index in the song
        # ----- find the number of the file -----
        file_is_not_found = True
        while file_is_not_found:
            if index + self.nb_elements_available_per_file[file] > i_start:
                # Then the file is long enough to contain the corresponding step
                file_is_not_found = False
            else:
                # The corresponding step is in another file
                index += self.nb_elements_available_per_file[file]
                file += 1
        # ----- Find the corresponding song -----
        song_is_not_found = True
        while song_is_not_found:
            if index + self.nb_elements_available_per_song[file][song] > i_start:
                song_is_not_found = False
            else:
                index += self.nb_elements_available_per_song[file][song]
                song += 1
        index = i_start - index
        return file, song, index

    def return_nb_elements_available(self):
        """

        :return: The number of useful element in the dataset

        A recursive function is used
        Go into every list and add up the (length // step_size) - (nb_steps - 1 + predict_offset)
        """

        def _return_nb_elements(l):
            """

            :param l:
            :return:
            """
            if type(l) is list:
                acc = 0
                for l2 in l:
                    acc += _return_nb_elements(l2)
                return acc
            else:
                return self.len_to_nb_available(l[0])

        return _return_nb_elements(self.all_shapes)

    def get_nb_elements_available_per_file(self):
        """

        :return: all_len = List(nb_files)[int]      the int is the number of steps available in the files
        """

        def f_map(l):
            return functools.reduce(lambda x, y: x + self.len_to_nb_available(y[0]),
                                    l,
                                    0)  # not -self.nb_steps + 1 because of the y (true tab)

        all_len = list(map(f_map, self.all_shapes))

        return all_len

    def get_nb_elements_available_per_song(self):
        """

        :return:
        """
        nb_elements_available_per_song = []
        for file in range(len(self.all_shapes)):
            nb_elements_available_per_song.append(
                list(map(
                    lambda x: self.len_to_nb_available(x[0]),
                    self.all_shapes[file]       # List(nb_songs_per_file)[(length, nb_instruments, input_size, channels)
                ))
            )
        return nb_elements_available_per_song

    def len_to_nb_available(self, length):
        """

        :param length:
        :return:
        """
        return (length // self.step_size) - (self.nb_steps + self.predict_offset) + 1

    def __del__(self, *args, **kwargs):
        del self.npy_loaded

    # ----------------------------------------------------------------------------------------------------
    #                                       To work with only one song
    # ----------------------------------------------------------------------------------------------------

    # The number of songs is available with the attribut self.nb_songs

    def song_file(self, song_number):
        """

        :param song_number:
        :return: the number of the file where the song is stored
        """
        file = 0
        acc = 0
        while acc + len(self.nb_elements_available_per_song[file]) < song_number:
            acc += len(self.nb_elements_available_per_song[file])
            file += 1
        return file

    def get_song_len(self, song_number):
        """

        :param song_number:
        :return: The number of steps available in this song
        """
        file_number = self.song_file(song_number)
        length = self.nb_elements_available_per_song[file_number][song_number]
        return length

    def get_index_first_step_song(self, song_number):
        """

        :param song_number:
        :return:
        """
        # Find back the index of the step for this song
        file_number = self.song_file(song_number)
        song_index = 0
        index = 0
        # First add all the steps of the preceding files
        for file in range(file_number):
            index += self.nb_elements_available_per_file[file]
            song_index += len(self.nb_elements_available_per_file[file])
        # Add the steps of all the preceding songs
        song_index = song_number - song_index
        for song in range(song_index):
            index += self.nb_elements_available_per_song[file_number][song]
        return index

    def get_song_step(self, song_number, step_number):
        """

        :param song_number:
        :param step_number:
        :return: The input and output for the step of this
        """
        old_batch_size = self.batch_size
        self.batch_size = 1
        index = self.get_index_first_step_song(song_number)
        index += step_number
        res = self.get_item(index)
        self.batch_size = old_batch_size
        return res

    def get_all_song(self, song_number, in_batch_format=True):
        """

        :param in_batch_format:
        :param song_number:
        :return:
        """
        init_params = self.get_init_params()
        init_params['replicate'] = True
        init_params['nb_steps'] = 1
        init_params['batch_size'] = 1
        sub_sequence = KerasSequence(**init_params)
        x_list = []
        y_list = []
        start_index = sub_sequence.get_index_first_step_song(song_number)
        for s in range(sub_sequence.get_song_len(song_number)):
            x, y = sub_sequence[start_index + s]
            # x (nb_instruments, batch=1, nb_steps, step_size, input_size, 2)
            # y (nb_instruments, batch=1, nb_steps=1, step_size, input_size, 2)
            # x can be with noise and y is without noise
            x_list.append(x)
            y_list.append(y)
        axis = 1 if in_batch_format else 2
        x = np.concatenate(x_list, axis=axis)
        y = np.concatenate(y_list, axis=axis)
        del sub_sequence
        return x, y





