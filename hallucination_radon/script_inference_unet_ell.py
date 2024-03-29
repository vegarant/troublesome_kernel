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
from operators import l2_error
import matplotlib.pyplot as plt


# ----- load configuration -----
import config  # isort:skip

# ----- global configuration -----
mpl.use("agg")
device = torch.device("cuda:0")
torch.cuda.set_device(0)
N = config.n[0]
## TD: Update
DATA_PATH = '/mn/kadingir/vegardantun_000000/nobackup/ellipses/raw_data_radon_pt' 
model_dir_name = 'model_unet_ell';
path_weights =  f"{config.RESULTS_PATH}/{model_dir_name}/train_phase_2/model_weights.pt"

with open(
    os.path.join(config.RESULTS_PATH, model_dir_name, "hyperparameters.yml"), 
                 "r") as file1:
    cgf = yaml.load(file1, Loader=yaml.UnsafeLoader)
subnet_params = cgf['SUBNET'];
it_net_params = cgf['IT_NET'];
subnet =  UNet 

print(f'PID: {os.getpid()}')
print('N: ', N)
# ----- data configuration -----

train_data_params = {
    "path_train": join(DATA_PATH, 'train'),
}
train_data = Load_radon_dataset

val_data_params = {
    "path_val": join(DATA_PATH, 'val'),
}
val_data = Load_radon_dataset
train_phases = len(cgf['TRAIN']['num_epochs']) 

# ------ construct network and train -----
subnet = subnet(**subnet_params).to(device)
it_net = IterativeNet(subnet, **it_net_params).to(device)

it_net.load_state_dict(
    torch.load(
        path_weights, 
    ),
    strict=False
)
it_net.freeze()
it_net.eval()

train_data = train_data("train", **train_data_params)
val_data = val_data("val", **val_data_params)

inp, tar = train_data[0];
inp = inp.to(device).unsqueeze(0)
tar = tar.to(device).unsqueeze(0)
print('inp.device: ', inp.device)
print('inp.shape: ', inp.shape)
print('inp.dtype: ', inp.dtype)


pred = it_net(inp);
print('pred.shape: ', pred.shape)
print('inp.shape: ', inp.shape)
print('tar.shape: ', tar.shape)

error= l2_error(pred, tar, relative=True, squared=False
                )[0].item()

print('error: ', error)


np_pred = pred.squeeze().cpu().detach().numpy()
np_inp = inp.squeeze().cpu().detach().numpy()
np_tar = tar.squeeze().cpu().detach().numpy()

plt.figure();

ax = plt.subplot(131);
plt.matshow(np_inp, cmap='gray', fignum=False);
ax.axis('off')
plt.title('Input');

ax = plt.subplot(132);
plt.matshow(np_pred, cmap='gray', fignum=False);
ax.axis('off')
plt.title('Reconstruction');

ax = plt.subplot(133);
plt.matshow(np_tar, cmap='gray', fignum=False);
ax.axis('off')
plt.title('Ground Truth');

plt.savefig('plots/rec_ell.pdf', bbox_inches='tight')



