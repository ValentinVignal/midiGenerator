import os

from .AEMono import AEMono as AEMono
from .AEMonoRep import AEMonoRep
from . import RMVAEMono
from src.NN import Sequences
from . import RRMVAEMono

from_name = dict(
    AEMono=AEMono.create_model,
    AEMonoRep=AEMonoRep.create_model,
    RMVAEMono=RMVAEMono.get_create(replicate=False, music=False),
    RMVAEMonoRep=RMVAEMono.get_create(replicate=True, music=False),
    MRMVAEMono=RMVAEMono.get_create(replicate=False, music=True),
    MRMVAEMonoRep=RMVAEMono.get_create(replicate=True, music=True),
    RRMVAEMono=RRMVAEMono.get_create(replicate=False, music=False),
    RRMVAEMonoReplicate=RRMVAEMono.get_create(replicate=True, music=False),
    MRRMVAEMono=RRMVAEMono.get_create(replicate=False, music=True),
    MRRMVAEMonoReplicate=RRMVAEMono.get_create(replicate=True, music=True),

)

param_folder_from_name = dict(
    AEMono='AEMono',
    AEMonoRep='AEMono',
    RMVAEMono=os.path.join('RMVAEMono', 'params'),
    RMVAEMonoRep=os.path.join('RMVAEMono', 'params'),
    MRMVAEMono=os.path.join('RMVAEMono', 'params'),
    MRMVAEMonoRep=os.path.join('RMVAEMono', 'params'),
    RRMVAEMono=os.path.join('RRMVAEMono', 'params'),
    RRMVAEMonoRep=os.path.join('RRMVAEMono', 'params'),
    MRRMVAEMono=os.path.join('RRMVAEMono', 'params'),
    MRRMVAEMonoRep=os.path.join('RRMVAEMono', 'params')
)

sequences = dict(
    AEMono=Sequences.AllInstSequence.getter(replicate=False),
    AEMonoRep=Sequences.AllInstSequence.getter(replicate=True),
    RMVAEMono=Sequences.MissingInstSequence.getter(replicate=False),
    RMVAEMonoRep=Sequences.MissingInstSequence.getter(replicate=True),
    MRMVAEMono=Sequences.MissingInstSequence.getter(replicate=False, scale=True),
    MRMVAEMonoRep=Sequences.MissingInstSequence.getter(replicate=True, scale=True),
    RRMVAEMono=Sequences.MissingInstSequence.getter(replicate=False),
    RRMVAEMonoRep=Sequences.MissingInstSequence.getter(replicate=True),
    MRRMVAEMono=Sequences.MissingInstSequence.getter(replicate=False, scale=True),
    MRRMVAEMonoRep=Sequences.MissingInstSequence.getter(replicate=True, scale=True)
)

needs_mask = dict(
    AEMono=False,
    AEMonoRep=False,
    RMVAEMono=True,
    RMVAEMonoRep=True,
    MRMVAEMono=True,
    MRMVAEMonoRep=True,
    RRMVAEMono=True,
    RRMVAEMonoRep=True,
    MRRMVAEMono=True,
    MRRMVAEMonoRep=True
)
