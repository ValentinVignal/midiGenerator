from .RMVAE import RMVAE as RMVAE
from .AEMono import AEMono as AEMono
from .AEMonoRep import AEMonoRep
from src.NN import Sequences

from_name = dict(
    RMVAE=RMVAE,
    AEMono=AEMono,
    AEMonoRep=AEMonoRep
)

sequences = dict(
    RMVAE=Sequences.MissingInstSequence,
    AEMono=Sequences.AllInstSequence,
    AEMonoRep=Sequences.AllInstSequenceReplicate
)

