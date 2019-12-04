from .KerasSequence import KerasSequence
from .AllInstSequence import AllInstSequence
from .AllInstSequenceReplicate import AllInstSequenceReplicate
from .MissingInstSequence import MissingInstSequence
from .TrainValSequence import ReducedSequence
from . import TrainValSequence

from_name = dict(
    KerasSequence=KerasSequence,
    AllInstSequence=AllInstSequence,
    AllInstSequenceReplicate=AllInstSequenceReplicate,
    MissingInstSequence=MissingInstSequence,
    ReducedSequence=ReducedSequence,
    TrainValSequence=TrainValSequence
)

