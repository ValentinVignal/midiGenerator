from .RMVAE import RMVAE as RMVAE
from .AEMono import AEMono as AEMono
from .AEMonoRep import AEMonoRep
from .PoEMono import PoEMono
from .PoEMonoRep import PoEMonoRep
from src.NN import Sequences

from_name = dict(
    RMVAE=RMVAE,
    AEMono=AEMono,
    AEMonoRep=AEMonoRep,
    PoEMono=PoEMono,
    PoEMonoRep=PoEMonoRep
)

sequences = dict(
    RMVAE=Sequences.MissingInstSequence.predict,
    AEMono=Sequences.AllInstSequence.predict,
    AEMonoRep=Sequences.AllInstSequence.replicate,
    PoEMono=Sequences.MissingInstSequence.predict,
    PoEMonoRep=Sequences.MissingInstSequence.replicate
)

needs_mask = dict(
    RMVAE=True,
    AEMono=False,
    AEMonoRep=False,
    PoEMono=True,
    PoEMonoRep=True
)

