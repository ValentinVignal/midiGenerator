from .RMVAE import RMVAE as RMVAE
from .AEMono import AEMono as AEMono

from_name = dict(
    RMVAE=RMVAE,
    AEMono=AEMono
)

sequences = dict(
    RMVAE='MissingInstSequence',
    AEMono='AllInstSequence'
)

