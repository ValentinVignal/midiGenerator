import numpy as np

import matplotlib.pyplot as plt

import src.image.pianoroll as pianoroll
from src.NN.Sequences.KerasSequence import KerasSequence


class AllInstSequenceFill(KerasSequence):
    def __init__(self, *args, **kwargs):
        super(AllInstSequenceFill, self).__init__(*args, fill=True, **kwargs)

    def __len__(self):
        return super(AllInstSequenceFill, self).__len__()

    def __getitem__(self, item):
        x, y = super(AllInstSequenceFill, self).__getitem__(item)
        return list(x), list(y)

