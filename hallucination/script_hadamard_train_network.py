import os

import matplotlib as mpl
import torch
import torchvision
import yaml

from extra_utils import read_sampling_pattern_from_file, read_count

from os.path import join
from data_management import IPDataset, Jitter, SimulateMeasurements 
from networks import IterativeNet, UNet, Tiramisu
from operators import (
    Hadamard,
    LearnableInverterHadamard,
)


# ----- load configuration -----
import config  # isort:skip

# ----- global configuration -----
mpl.use("agg")
device = torch.device("cuda:0")
torch.cuda.set_device(0)
N = config.n[0]
learnableInv = True
src_patt = './samp_patt'

# ----- measurement configuration -----

# Pattern parameters
nbr_levels = 50
a = 1.75
r0 = 0
srate = 0.05


samp_patt_name = join(
    config.src_patt,
    f"sph2_cgauss_N_{N}_srate_{round(100*srate):02d}_r_{nbr_levels}_r0_{r0}_a_{round(100*a)}.png"
)
print(samp_patt_name)
mask = read_sampling_pattern_from_file(samp_patt_name)
print(f'PID: {os.getpid()}')
print('N: ', N)
print('a: ', a)
print('r0: ', r0)
print('srate: ', srate)

OpA = Hadamard(mask)
inverter = LearnableInverterHadamard(config.n, mask, learnable=learnableInv)

print(inverter)

# ----- network configuration -----
subnet_params = {
    "in_channels": 1,
    "out_channels": 1,
    #"drop_factor": 0.0,
    #"base_features": 32,
    "down_blocks": (5, 7, 9, 12, 15),
    "up_blocks": (15, 12, 9, 7, 5),
    "pool_factors": (2, 2, 2, 2, 2),
    "bottleneck_layers": 20,
    "growth_rate": 16,
    "out_chans_first_conv": 16,
}
subnet =  Tiramisu

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


model_nbr = read_count('./')
model_dir_name = f'model_{model_nbr:03d}' 
print('Model number: ', model_nbr)
if not os.path.isdir(config.RESULTS_PATH):
    os.mkdir(config.RESULTS_PATH)
if not os.path.isdir(join(config.RESULTS_PATH, model_dir_name)):
    os.mkdir(join(config.RESULTS_PATH, model_dir_name))

train_phases = 2
train_params = {
    "num_epochs": [15, 10],
    "batch_size": [2, 2],
    "loss_func": loss_func,
    "save_path": [
        os.path.join(
            config.RESULTS_PATH,
            model_dir_name,
            "train_phase_{}".format(i + 1),
        )
        for i in range(train_phases)
    ],
    "save_epochs": 5,
    "optimizer": torch.optim.Adam,
    "optimizer_params": [
        {"lr": 2e-4, "eps": 1e-5, "weight_decay": 1e-3},
        {"lr": 5e-5, "eps": 1e-5, "weight_decay": 5e-4},
    ],
    "scheduler": torch.optim.lr_scheduler.StepLR,
    "scheduler_params": {"step_size": 1, "gamma": 1.0},
    "acc_steps": [1, 200],
    "train_transform": torchvision.transforms.Compose(
        [SimulateMeasurements(OpA), Jitter(1e1, 0.0, 1.0)]
    ),
    "val_transform": torchvision.transforms.Compose(
        [SimulateMeasurements(OpA)],
    ),
    "train_loader_params": {"shuffle": True, "num_workers": 8},
    "val_loader_params": {"shuffle": False, "num_workers": 8},
}

# ----- data configuration -----

train_data_params = {
    "path": config.DATA_PATH,
}
train_data = IPDataset

val_data_params = {
    "path": config.DATA_PATH,
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
                'a': a, 
                'r0': r0, 
                'fname_patt': samp_patt_name, 
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
