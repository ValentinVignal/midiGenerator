import numpy as np

import matplotlib.pyplot as plt

import src.image.pianoroll as pianoroll
from src.NN.Sequences.KerasSequence import KerasSequence


class AllInstSequenceReplicate(KerasSequence):
    def __init__(self, *args, **kwargs):
        super(AllInstSequenceReplicate, self).__init__(*args, replicate=True, **kwargs)

    def __len__(self):
        return super(AllInstSequenceReplicate, self).__len__()

    def __getitem__(self, item):
        x, y = super(AllInstSequenceReplicate, self).__getitem__(item)
        return list(x), list(y)

