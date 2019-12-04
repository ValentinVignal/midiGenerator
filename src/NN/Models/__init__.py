from .RMVAE import RMVAE as RMVAE
from .AEMono import AEMono as AEMono
from .AEMonoFill import AEMonoFill
from src.NN import Sequences

from_name = dict(
    RMVAE=RMVAE,
    AEMono=AEMono,
    AEMonoFill=AEMonoFill
)

sequences = dict(
    RMVAE=Sequences.MissingInstSequence,
    AEMono=Sequences.AllInstSequence,
    AEMonoFill=Sequences.AllInstSequenceFill
)

