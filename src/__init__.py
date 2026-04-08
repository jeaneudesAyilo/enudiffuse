from sgmse.util.registry import Registry

InferenceAlgoRegistry = Registry("InferenceAlgo")

from .udiffse import UDiffSE
from .fudiffse import fUDiffSE
from .depse_il import DEPSE_IL
from .depse_tl import DEPSE_TL
from .diffuseen import DiffUSEEN
from .joint_paradiffusein import JointParaDiffUSEIN
from .joint_paradiffuseen import JointParaDiffUSEEN
from .separate_paradiffusein import SeparateParaDiffUSEIN
from .separate_paradiffuseen import SeparateParaDiffUSEEN


__all__ = ['InferenceAlgoRegistry', 'UDiffSE','fUDiffSE', 'DEPSE_IL', 'DEPSE_TL', 
            'DiffUSEEN', 'JointParaDiffUSEIN', 'JointParaDiffUSEEN', 
            'SeparateParaDiffUSEIN', 'SeparateParaDiffUSEEN']
