import os

from data_management import (
    Load_radon_dataset,
)

RESULTS_PATH = os.path.join("models")

# ----- random seeds -----
torch_seed = 1
numpy_seed = 2
matrix_seed = 3

# ----- signal configuration -----

n = (256, 256) # signal dimension
src_patt = './samp_patt'
DATA_PATH =  '/mn/kadingir/vegardantun_000000/nobackup/pytorch_datasets/BioSR/ER/res_512'


data_params = {
    'path_train' : '/mn/kadingir/vegardantun_000000/nobackup/BioSR/ER_generated_data/res_256/train',
    'path_test' : '/mn/kadingir/vegardantun_000000/nobackup/BioSR/ER_generated_data/res_256/test',
    'path_val' : '/mn/kadingir/vegardantun_000000/nobackup/BioSR/ER_generated_data/res_256/val'
}

'''
data_params = {
    'path_train' : '/mn/kadingir/vegardantun_000000/nobackup/fastMRI/multicoil_train',
    'path_val' : '/mn/kadingir/vegardantun_000000/nobackup/fastMRI/multicoil_val_small'
}
'''

data_gen = Load_radon_dataset

# ----- data set configuration -----
set_params = {
    "path": DATA_PATH,
}
