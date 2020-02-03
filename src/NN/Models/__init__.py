from .AEMono import AEMono as AEMono
from .AEMonoRep import AEMonoRep
from .RMVAEMono import RMVAEMono
from src.NN import Sequences

from_name = dict(
    AEMono=AEMono.create_model,
    AEMonoRep=AEMonoRep.create_model,
    RMVAEMono=RMVAEMono.create_model,
    RMVAEMonoRep=RMVAEMono.create_model_rep
)

param_folder_from_name = dict(
    AEMono='AEMono',
    AEMonoRep='AEMono',
    RMVAEMono='RMVAEMono',
    RMVAEMonoRep='RMVAEMono'
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

