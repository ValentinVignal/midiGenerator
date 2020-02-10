import os

from .AEMono import AEMono as AEMono
from .AEMonoRep import AEMonoRep
from . import RMVAEMono
from src.NN import Sequences

from_name = dict(
    AEMono=AEMono.create_model,
    AEMonoRep=AEMonoRep.create_model,
    RMVAEMono=RMVAEMono.get_create(replicate=False, scale=False),
    RMVAEMonoRep=RMVAEMono.get_create(replicate=True, scale=False),
    RMVAEScaleMono=RMVAEMono.get_create(replicate=False, scale=True),
    RMVAEScaleMonoRep=RMVAEMono.get_create(replicate=True, scale=True)

)

param_folder_from_name = dict(
    AEMono='AEMono',
    AEMonoRep='AEMono',
    RMVAEMono=os.path.join('RMVAEMono', 'params'),
    RMVAEMonoRep=os.path.join('RMVAEMono', 'params'),
    RMVAEScaleMono=os.path.join('RMVAEMono', 'params'),
    RMVAEScaleMonoRep=os.path.join('RMVAEMono', 'params')
)

sequences = dict(
    AEMono=Sequences.AllInstSequence.getter(replicate=False),
    AEMonoRep=Sequences.AllInstSequence.getter(replicate=True),
    RMVAEMono=Sequences.MissingInstSequence.getter(replicate=False),
    RMVAEMonoRep=Sequences.MissingInstSequence.getter(replicate=True),
    RMVAEScaleMono=Sequences.MissingInstSequence.getter(replicate=False, scale=True),
    RMVAEScaleMonoRep=Sequences.MissingInstSequence.getter(replicate=True, scale=True)
)

needs_mask = dict(
    AEMono=False,
    AEMonoRep=False,
    RMVAEMono=True,
    RMVAEMonoRep=True
)

