import numpy as np

from src.NN.sequences.KerasSequence import KerasSequence


class MissingInstSequence(KerasSequence):
    """
    train on:
        only one expert
        everyone
        k random combinations
    """
    def __init__(self, k=2, batch_size=4, *args, **kwargs):
        super(MissingInstSequence, self).__init__(*args, batch_size=1, **kwargs)

        self.k = k
        self.batch_size = batch_size
        self.nb_instruments = super(MissingInstSequence, self).__getitem__(0)[0][0].shape[0]
        self.nb_combinations = self.nb_instruments + 1 + k

    def __len__(self):
        return (super(MissingInstSequence, self).__len__() * self.nb_combinations) // self.batch_size

    def __getitem__(self, item):
        x_batch, y_batch = [], []
        i_start = item * self.batch_size
        for b in range(self.batch_size):
            x, y = super(MissingInstSequence, self).__getitem__((i_start + b) // self.nb_combinations)
            # x (nb_instruments, batch, nb_steps, step_size, input_size, 2)
            # x (nb_instruments, batch, nb_steps=1, step_size, input_size, 2)
            mod = item % self.nb_combinations
            if mod < self.nb_instruments:
                # Only on instrument
                x[:mod] = np.nan
                x[mod+1:] = np.nan
                y[:mod] = np.nan
                y[mod+1:] = np.nan
            elif mod == self.nb_instruments:
                # All the instruments
                pass
            else:
                # Combinaison
                nan_axis = np.random.choice(
                    a=self.nb_instruments,
                    size=np.random.randint(1, self.nb_instruments-1),
                    replace=False)
                x[nan_axis] = np.nan
                y[nan_axis] = np.nan
                pass
            x_batch.append(x)
            y_batch.append(y)
        x = np.concatenate(x_batch, axis=1)
        y = np.concatenate(y_batch, axis=1)
        return list(x), list(y)
