import numpy as np

import matplotlib.pyplot as plt

import src.image.pianoroll as pianoroll
from src.NN.Sequences.KerasSequence import KerasSequence


class AllInstSequence(KerasSequence):
    def __init__(self, *args, **kwargs):
        super(AllInstSequence, self).__init__(*args, **kwargs)

    def __len__(self):
        return super(AllInstSequence, self).__len__()

    def __getitem__(self, item):
        x, y = super(AllInstSequence, self).__getitem__(item)
        return list(x), list(y)


# ------------------------------------------------------------


class SeeAllInstSequence:

    def __init__(self, path, nb_steps, work_on):
        self.my_sequence = AllInstSequence(path=path, nb_steps=nb_steps, batch_size=1, work_on=work_on)
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


