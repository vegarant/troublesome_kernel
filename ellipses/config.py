import os

from data_management import sample_ellipses

N = 256;

DATA_PATH = os.path.join("/mn/kadingir/vegardantun_000000/nobackup/ellipses/raw_data")
RESULTS_PATH = os.path.join(f"models")

# ----- random seeds -----
torch_seed = 1
numpy_seed = 2
matrix_seed = 3

# Sampling pattern
use_pattern_from_file = False
fname_patt = f'/mn/sarpanitu/ansatte-u4/vegarant/storage_stable_NN/samp_patt/XXX.png'

# ----- signal configuration -----
n = (N, N)  # signal dimension
data_params = {  # additional data generation parameters
    "c_min": 10,
    "c_max": 40,
    "max_axis": 0.15,
    "min_axis": 0.01,
    "margin_offset": 0.3,
    "margin_offset_axis": 0.9,
    "grad_fac": 0.9,
    "bias_fac": 1.0,
    "bias_fac_min": 0.3,
    "normalize": True,
}
data_gen = sample_ellipses  # data generator function

# ----- data set configuration -----
set_params = {
    "num_train": 25000,
    "num_val": 1000,
    "num_test": 1000,
    "path": DATA_PATH,
}
