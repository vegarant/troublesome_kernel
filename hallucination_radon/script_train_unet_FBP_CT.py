import os

import matplotlib as mpl
import torch
import torchvision
import yaml

from extra_utils import (
    read_count, 
)

from os.path import join
from data_management import Load_radon_dataset 
from networks import IterativeNet, UNet, Tiramisu

# ----- load configuration -----
import config  # isort:skip

# ----- global configuration -----
mpl.use("agg")
device = torch.device("cuda:0")
torch.cuda.set_device(0)
N = config.n[0]

DATA_PATH = '/mn/kadingir/vegardantun_000000/nobackup/CT_images/pt_files' 

# ----- measurement configuration -----
inverter = torch.nn.Identity()

print(f'PID: {os.getpid()}')
print('N: ', N)

# ----- network configuration -----
subnet =  UNet 

# ----- training configuration -----
mseloss = torch.nn.MSELoss(reduction="sum")

def loss_func(pred, tar):
    
    return (
        mseloss(pred, tar)
        / pred.shape[0]
    )


model_dir_name = f'model_unet_CT' 
if not os.path.isdir(config.RESULTS_PATH):
    os.mkdir(config.RESULTS_PATH)
if not os.path.isdir(join(config.RESULTS_PATH, model_dir_name)):
    os.mkdir(join(config.RESULTS_PATH, model_dir_name))

train_phases = 2
train_params = {
    "num_epochs": [500,20], #[500, 100],
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
    "save_epochs": 10, # 50
    "optimizer": torch.optim.Adam,
    "optimizer_params": [
        {"lr": 2e-4, "eps": 1e-5, "weight_decay": 1e-3},
        {"lr": 5e-5, "eps": 1e-5, "weight_decay": 5e-4},
    ],
    "scheduler": torch.optim.lr_scheduler.StepLR,
    "scheduler_params": {"step_size": 1, "gamma": 1.0},
    "acc_steps": [1, 200],
    "train_transform": None,
    "val_transform": None, 
    "train_loader_params": {"shuffle": True, "num_workers": 8},
    "val_loader_params": {"shuffle": False, "num_workers": 8},
}

# ----- data configuration -----

train_data_params = {
    "path_train": join(DATA_PATH, 'train'),
}
train_data = Load_radon_dataset

val_data_params = {
    "path_val": join(DATA_PATH, 'val'),
}
val_data = Load_radon_dataset

# ------ save hyperparameters -------
#os.makedirs(train_params["save_path"][-1], exist_ok=True)

fname_hyperparam = 'models/model_unet_ell/hyperparameters.yml'
with open(fname_hyperparam, 'r') as file1:
    cgf_old_model = yaml.load(file1, Loader=yaml.UnsafeLoader)

it_net_params = cgf_old_model['IT_NET']
subnet_params = cgf_old_model['SUBNET']

cgf = {'SUBNET': subnet_params,
       'IT_NET': it_net_params,
       'TRAIN': train_params.copy(),
       'TRAIN_DATA': train_data_params,
       'VAL_DATA': val_data_params,
       'DATA': {'N': N}, 
        }

cgf['TRAIN']['scheduler'] = str(cgf['TRAIN']['scheduler'])
cgf['TRAIN']['train_transform'] = str(cgf['TRAIN']['train_transform'])
cgf['TRAIN']['val_transform'] = str(cgf['TRAIN']['val_transform'])
cgf['TRAIN']['optimizer'] = str(cgf['TRAIN']['optimizer'])
cgf['TRAIN']['loss_func'] = None

with open(
    os.path.join(config.RESULTS_PATH, model_dir_name, "hyperparameters.yml"), "w"
) as file1:
    yaml.dump(cgf, file1)


# ------ construct network and train -----
subnet = subnet(**subnet_params).to(device)
it_net = IterativeNet(subnet, **it_net_params).to(device)

it_net.load_state_dict(
    torch.load(
        f"{config.RESULTS_PATH}/model_unet_ell/train_phase_2/model_weights.pt",
        map_location=torch.device(device),    
    ),
    strict=False
)

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




