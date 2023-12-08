import os

import matplotlib as mpl
import torch
import torchvision
import yaml

from extra_utils import (
    read_sampling_pattern_from_file, 
    read_count, 
    nullspace_proj
)

from os.path import join
from data_management import IPDataset, Jitter, SimulateMeasurements 
from networks import IterativeNet, UNet, Tiramisu
from operators import (
    rotate_real,
    Fourier,
    LearnableInverterFourier
)


# ----- load configuration -----
import config  # isort:skip

# ----- global configuration -----
mpl.use("agg")
device = torch.device("cuda:0")
torch.cuda.set_device(0)
N = config.n[0]
learnableInv = False
src_patt = './samp_patt'
DATA_PATH =  '/mn/kadingir/vegardantun_000000/nobackup/pytorch_datasets/fastMRI_complex'
srate = 0.2
nbr_levels = 50
a = 1.75
r0 = 0

# ----- measurement configuration -----

fname_patt = join(src_patt, f'spf2_cgauss_N_{N}_srate_{int(100*srate):02d}_r_{nbr_levels}_r0_{r0}_a_{int(100*a)}.png')
mask = read_sampling_pattern_from_file(fname_patt);
print(f'PID: {os.getpid()}')
print('N: ', N)
print('srate: ', srate)

OpA = Fourier(mask)
inverter = LearnableInverterFourier(config.n, mask, learnable=learnableInv)

print(inverter)

# ----- network configuration -----
subnet_params = {
    "in_channels": 2,
    "out_channels": 2,
    "drop_factor": 0.0,
    "base_features": 32,
}
subnet =  UNet 

it_net_params = {
    "num_iter": 1,
    "lam": 0.1,
    "lam_learnable": True,
    "final_dc": False,
    "resnet_factor": 1.0,
    "operator": OpA,
    "inverter": inverter,
}

# ----- training configuration -----
mseloss = torch.nn.MSELoss(reduction="sum")


def loss_func(pred, tar):
    
    return (
        mseloss(pred, tar)
        / pred.shape[0]
    )


model_nbr = 10; #read_count('./')
model_dir_name = f'model_{model_nbr:03d}' 
print('Model number: ', model_nbr)
if not os.path.isdir(config.RESULTS_PATH):
    os.mkdir(config.RESULTS_PATH)
if not os.path.isdir(join(config.RESULTS_PATH, model_dir_name)):
    os.mkdir(join(config.RESULTS_PATH, model_dir_name))

train_phases = 2
train_params = {
    "num_epochs": [500, 100],
    "batch_size": [10, 10],
    "loss_func": loss_func,
    "save_path": [
        os.path.join(
            config.RESULTS_PATH,
            model_dir_name,
            "train_phase_{}".format(i + 1),
        )
        for i in range(train_phases)
    ],
    "save_epochs": 50,
    "optimizer": torch.optim.Adam,
    "optimizer_params": [
        {"lr": 2e-4, "eps": 1e-5, "weight_decay": 1e-3},
        {"lr": 5e-5, "eps": 1e-5, "weight_decay": 5e-4},
    ],
    "scheduler": torch.optim.lr_scheduler.StepLR,
    "scheduler_params": {"step_size": 1, "gamma": 1.0},
    "acc_steps": [1, 200],
    "train_transform": [
        torchvision.transforms.Compose(
            [SimulateMeasurements(OpA), Jitter(1e1, 0.0, 1.0)]
        ), 
        torchvision.transforms.Compose(
            [SimulateMeasurements(OpA)]
        ), 
    ],
    "val_transform": torchvision.transforms.Compose(
        [SimulateMeasurements(OpA)],
    ),
    "train_loader_params": {"shuffle": True, "num_workers": 8},
    "val_loader_params": {"shuffle": False, "num_workers": 8},
}

# ----- data configuration -----

train_data_params = {
    "path": DATA_PATH,
}
train_data = IPDataset

val_data_params = {
    "path": DATA_PATH,
}
val_data = IPDataset

# ------ save hyperparameters -------
#os.makedirs(train_params["save_path"][-1], exist_ok=True)

cgf = {'SUBNET': subnet_params,
       'IT_NET': it_net_params.copy(),
       'TRAIN': train_params.copy(),
       'TRAIN_DATA': train_data_params,
       'VAL_DATA': val_data_params,
       'DATA': {'N': N, 
                'srate': srate, 
                'fname_patt': fname_patt, 
                'subnet' : str(subnet),
                'learnableInv': learnableInv}
        }

cgf['TRAIN']['scheduler'] = str(cgf['TRAIN']['scheduler'])
cgf['TRAIN']['train_transform'] = str(cgf['TRAIN']['train_transform'])
cgf['TRAIN']['val_transform'] = str(cgf['TRAIN']['val_transform'])
cgf['TRAIN']['optimizer'] = str(cgf['TRAIN']['optimizer'])
cgf['TRAIN']['loss_func'] = None
cgf['IT_NET']['operator'] = str(cgf['IT_NET']['operator'])
cgf['IT_NET']['inverter'] = str(cgf['IT_NET']['inverter'])

with open(
    os.path.join(config.RESULTS_PATH, model_dir_name, "hyperparameters.yml"), "w"
) as file1:
    yaml.dump(cgf, file1)

# ------ construct network and train -----
subnet = subnet(**subnet_params).to(device)
it_net = IterativeNet(subnet, **it_net_params).to(device)
train_data = train_data("train", **train_data_params)
val_data = val_data("val", **val_data_params)

for i in range(train_phases):
    train_params_cur = {}
    for key, value in train_params.items():
        train_params_cur[key] = (
            value[i] if isinstance(value, (tuple, list)) else value
        )

    print("Phase {}:".format(i + 1))
    for key, value in train_params_cur.items():
        print(key + ": " + str(value))

    it_net.train_on(train_data, val_data, **train_params_cur)
