import numpy as np

from src.NN.Sequences.KerasSequence import KerasSequence


class MissingInstSequence(KerasSequence):
    """
    train on:
        only one expert
        everyone
        k random combinations
    """
    def __init__(self, *args, k=2, replicate=False, scale=False, **kwargs):
        super(MissingInstSequence, self).__init__(*args, replicate=replicate, **kwargs)

        self.k = k
        self.nb_combinations = min(self.nb_instruments + 1 + k, 2 ** self.nb_instruments - 1)
        self.scale = scale

    def __len__(self):
        return super(MissingInstSequence, self).__len__() * self.nb_combinations

    def __getitem__(self, item):
        """

        :param item:
        :return: x + [mask], y
            x: List(nb_instruments)[(batch, nb_steps, step_size, input_size, channels)]
            mask: (batch, nb_instruments, nb_steps)
            y: List(nb_instruments)[(batch, nb_steps, step_size, input_size, channels)]
        """
        x, y = super(MissingInstSequence, self).__getitem__(item // self.nb_combinations)
        # x (nb_instruments, batch, nb_steps, step_size, input_size, 2)
        # y (nb_instruments, batch, nb_steps=1, step_size, input_size, 2)
        # mask (batch, nb_instruments, nb_steps)
        mod = item % self.nb_combinations
        if mod == 0:
            # All the instruments
            mask = np.ones((self.batch_size, self.nb_instruments, self.nb_steps))
        elif mod < self.nb_instruments + 1:
            # Only on instrument
            mod -= 1
            mask = np.zeros((self.batch_size, self.nb_instruments, self.nb_steps))
            mask[:, mod] = 1
            y[mod:] = np.nan
            y[:mod + 1] = np.nan
        else:
            # Combinations
            zeros_axis = np.random.choice(
                a=self.nb_instruments,
                size=np.random.randint(1, self.nb_instruments-2),
                replace=False)
            mask = np.ones((self.batch_size, self.nb_instruments, self.nb_steps))
            mask[:, zeros_axis] = 0
            y[zeros_axis] = np.nan
        y_list = list(y)
        if self.scale:
            y_list.append(np.transpose(y, axes=[1, 0, 2, 3, 4, 5]))
            # y: (batch, nb_instruments, nb_steps, step_size, input_size, channels)
        return list(x) + [mask], y_list

    def get_song_step(self, song_number, step_number, with_mask=True, with_last_tensor=False):
        """

        :param with_last_tensor:
        :param song_number:
        :param step_number:
        :param with_mask:
        :return:
        """
        x, y = super(MissingInstSequence, self).get_song_step(song_number, step_number)
        # x (nb_instruments batch, nb_steps, step_size, input_size, channels)
        # y (nb_instruments batch, nb_steps, step_size, input_size, channels)
        batch_size = x.shape[1]
        x_list = list(x)
        y_list = list(y)
        if with_mask:
            mask = np.ones((batch_size, self.nb_instruments, self.nb_steps))
            x_list.append(mask)
        if with_last_tensor:
            y_list.append(np.transpose(y, [1, 0, 2, 3, 4, 5]))
        return x_list, y_list

    def get_all_song(self, song_number, in_batch_format=True):
        """

        :param song_number:
        :param in_batch_format:
        :param with_mask:
        :return:
        """
        x, y = super(MissingInstSequence, self).get_all_song(
            song_number=song_number,
            in_batch_format=in_batch_format
        )
        return list(x), list(y)

    @classmethod
    def predict(cls, *args, **kwargs):
        return cls(*args, replicate=False, **kwargs)

    @classmethod
    def replicate(cls, *args, **kwargs):
        return cls(*args, replicate=True, **kwargs)
