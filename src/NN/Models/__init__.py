from .RMVAE import RMVAE as RMVAE
from .AEMono import AEMono as AEMono
from src.NN import Sequences

from_name = dict(
    RMVAE=RMVAE,
    AEMono=AEMono
)

sequences = dict(
    RMVAE=Sequences.MissingInstSequence,
    AEMono=Sequences.AllInstSequence
)

