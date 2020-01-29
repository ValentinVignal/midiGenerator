# Keras Sequence
from .KerasSequence import KerasSequence
# Usable Sequences
from .AllInstSequence import AllInstSequence
from .MissingInstSequence import MissingInstSequence
from .TrainValSequence import ReducedSequence
# Wrappers
from . import TrainValSequence
from .to_numpy import sequence_to_numpy
from .FastSequence import FastSequence

from_name = dict(
    KerasSequence=KerasSequence,
    AllInstSequence=AllInstSequence,
    MissingInstSequence=MissingInstSequence,
    ReducedSequence=ReducedSequence,
    TrainValSequence=TrainValSequence
)



