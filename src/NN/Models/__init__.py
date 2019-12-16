from .AEMono import AEMono as AEMono
from .AEMonoRep import AEMonoRep
from .RMVAEMono import RMVAEMono
from .RMVAEMonoRep import RMVAEMonoRep
from src.NN import Sequences

from_name = dict(
    AEMono=AEMono,
    AEMonoRep=AEMonoRep,
    RMVAEMono=RMVAEMono,
    RMVAEMonoRep=RMVAEMonoRep
)

sequences = dict(
    AEMono=Sequences.AllInstSequence.predict,
    AEMonoRep=Sequences.AllInstSequence.replicate,
    RMVAEMono=Sequences.MissingInstSequence.predict,
    RMVAEMonoRep=Sequences.MissingInstSequence.replicate
)

needs_mask = dict(
    AEMono=False,
    AEMonoRep=False,
    RMVAEMono=True,
    RMVAEMonoRep=True
)

