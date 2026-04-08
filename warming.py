### assess if everything is working after installing the environment and downloading the checkpoints

from src import InferenceAlgoRegistry
DiffUSEEN = InferenceAlgoRegistry.get_by_name("diffuseen")

CKPT_PATH="./ckpts/separate_vbdmd_speech_modeling.ckpt"

num_E = 30 
verbose = True

diffuseen = DiffUSEEN(ckpt_path=CKPT_PATH, num_E=num_E, verbose=verbose)

_, _ = diffuseen.prior_sampler(clean_file = None, vfile_path = None)