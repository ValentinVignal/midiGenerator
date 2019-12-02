from .KerasSequence import KerasSequence
from .AllInstSequence import AllInstSequence
from .MissingInstSequence import MissingInstSequence
from .TrainValSequence import ReducedSequence
from . import TrainValSequence

from_name = dict(
    KerasSequence=KerasSequence,
    AllInstSequence=AllInstSequence,
    MissingInstSequence=MissingInstSequence,
    ReducedSequence=ReducedSequence,
    TrainValSequence=TrainValSequence
)

