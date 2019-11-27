import numpy as np

from src.NN.Sequences.KerasSequence import KerasSequence


class MissingInstSequence(KerasSequence):
    """
    train on:
        only one expert
        everyone
        k random combinations
    """
    def __init__(self, k=2, *args, **kwargs):
        super(MissingInstSequence, self).__init__(*args, **kwargs)

        self.batch_size = kwargs['batch_size']
        self.nb_steps = kwargs['nb_steps']
        self.k = k
        self.nb_instruments = super(MissingInstSequence, self).__getitem__(0)[0][0].shape[0]
        self.nb_combinations = self.nb_instruments + 1 + k

    def __len__(self):
        return super(MissingInstSequence, self).__len__()# * self.nb_combinations

    def __getitem__(self, item):
        x, y = super(MissingInstSequence, self).__getitem__(item)# // self.nb_combinations)
        '''
        # x (nb_instruments, batch, nb_steps, step_size, input_size, 2)
        # x (nb_instruments, batch, nb_steps=1, step_size, input_size, 2)
        mod = item % self.nb_combinations
        if mod < self.nb_instruments:
            # Only on instrument
            mask = np.zeros((self.batch_size, self.nb_instruments, self.nb_steps))
            mask[:, mod] = 1
            y[:, :mod] = np.nan
            y[:, :mod+1] = np.nan
        elif mod == self.nb_instruments:
            # All the instruments
            mask = np.ones((self.batch_size, self.nb_instruments, self.nb_steps))
        else:
            # Combinations
            zeros_axis = np.random.choice(
                a=self.nb_instruments,
                size=np.random.randint(1, self.nb_instruments-2),
                replace=False)
            mask = np.ones((self.batch_size, self.nb_instruments, self.nb_steps))
            mask[:, zeros_axis] = 0
            y[:, zeros_axis] = np.nan
        '''
        return list(x) + [np.ones((self.batch_size, self.nb_instruments, self.nb_steps))], list(y)

    def change_batch_size(self, batch_size):
        self.batch_size = batch_size
        super(MissingInstSequence, self).change_batch_size(batch_size)
