from .RMVAE import RMVAE as RMVAE
from .AEMono import AEMono as AEMono
from .AEMonoRep import AEMonoRep
from .PoEMonoRep import PoEMonoRep
from src.NN import Sequences

from_name = dict(
    RMVAE=RMVAE,
    AEMono=AEMono,
    AEMonoRep=AEMonoRep,
    PoEMonoRep=PoEMonoRep
)

sequences = dict(
    RMVAE=Sequences.MissingInstSequence.predict,
    AEMono=Sequences.AllInstSequence.predict,
    AEMonoRep=Sequences.AllInstSequence.replicate,
    PoEMonoRep=Sequences.MissingInstSequence.replicate
)

