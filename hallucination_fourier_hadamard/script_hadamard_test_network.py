import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
import yaml

from os.path import join
from shutil import copyfile

import matplotlib.pyplot as plt

import config

from extra_utils import read_sampling_pattern_from_file, cut_to_01

from networks import IterativeNet, Tiramisu, UNet
from operators import (
    Hadamard,
    LearnableInverterHadamard,
)


# ------ setup ----------
device = torch.device("cuda:0")
torch.cuda.set_device(0)

model_nbr = 82
epoch_nbr = -1
train_phase = 2
nbr_of_images = 100
dest_plots = 'plots'

if not os.path.isdir(dest_plots):
    os.mkdir(dest_plots)

config_file = join(config.RESULTS_PATH, f'model_{model_nbr:03d}', "hyperparameters.yml")
copyfile(config_file, join(dest_plots, f'hyp_mod_{model_nbr:03d}.yml'))

with open(config_file) as ymlfile:
    cgf = yaml.load(ymlfile)#, Loader=yaml.SafeLoader)

N = cgf['DATA']['N']
fname_patt = cgf['DATA']['fname_patt']
a  = cgf['DATA']['a']
r0 = cgf['DATA']['r0']
r = 50
subnet_name = cgf['DATA']['subnet']
learnableInv = cgf['DATA']['learnableInv']

print(f'PID: {os.getpid()}')
print(f'Model number: {model_nbr}')
print(f'N = {N}')
print(f'a = {a}')
print(f'r0 = {r0}')
print(f'r = 50')
print(f'SUBNET: {subnet_name}, {type(subnet_name)}')

if subnet_name == "<class 'networks.UNet'>":
    subnet = UNet
elif subnet_name == "<class 'networks.Tiramisu'>":
    subnet = Tiramisu
else:
    print('Error: subnet {subnet_name} is unknown')

n = (N, N)
mask = read_sampling_pattern_from_file(fname_patt).to(device);

# ----- operators -----
OpA = Hadamard(mask, device=device)

subnet_params = cgf['SUBNET'].copy()
it_net_params = cgf['IT_NET'].copy()
it_net_params['operator'] = OpA
inverter = LearnableInverterHadamard(n, mask, learnable=learnableInv).to(device)
#if it_net_params['inverter'].lower() == 'CheapNonLearnableInverter'.lower():
#   it_net_params['inverter'] = CheapNonLearnableInverter(OpA)
#else:
#    print('Error: Not implemented yet')
it_net_params['inverter'] = inverter
# loss
mseloss = torch.nn.MSELoss(reduction="sum")

# ----- load nets -----

# create a net and load weights from file
def _load_net(path, subnet, subnet_params, it_net_params):
    subnet = subnet(**subnet_params).to(device)
    it_net = IterativeNet(subnet, **it_net_params).to(device)
    it_net.load_state_dict(torch.load(path, map_location=torch.device(device)))
    it_net.freeze()
    it_net.eval()
    return it_net

#def _append_net(name, info, net):
#    methods.loc[name] = {
#        "info": info,
#        "reconstr": lambda y, noise_rel: _reconstructNet(y, noise_rel, net),
#        "attacker": lambda x0, noise_rel, yadv_init=None: _attackerNet(
#            x0, noise_rel, net, yadv_init=yadv_init
#        ),
#        "net": net,
#    }
#    pass

net_location = join(config.RESULTS_PATH, 
    f'model_{model_nbr:03d}', 
    f'train_phase_{train_phase}'
) 
if epoch_nbr > 0:
    epoch_name = f'model_weights_epoch{epoch_nbr:03d}.pt'
else: 
    epoch_name = "model_weights.pt"

net = _load_net(
    join(net_location, epoch_name),
    subnet,
    subnet_params,
    it_net_params
    )
print('Network loaded')
dirs = ['val']
split_loc = 'splits'
bd = 5
for dir1 in dirs:
    image_location = join(dest_plots, dir1)
    split_location = join(image_location, split_loc)
    if not os.path.isdir(image_location):
        os.mkdir(image_location)
    if not os.path.isdir(split_location):
        os.mkdir(split_location)
    for i in range(nbr_of_images):
        sample = torch.load(join(config.DATA_PATH, dir1, f'sample_{i:05d}.pt'))
        sample1 = sample.unsqueeze(0);
        sample1 = sample1.unsqueeze(0);
        X_0 = sample1.to(device)
        it_init = 1
        #print('hei: ', (X_0.ndim-1)*(1,))
        #X_0 = X_0.repeat(it_init, *((X_0.ndim - 1) * (1,)))
        #print('X_0.shape: ', X_0.shape)
        Y_0 = OpA(X_0)

        adj = OpA.adj(Y_0)
        adj_plot = adj.detach().cpu().numpy() 
        adj_plot = np.squeeze(adj_plot)
        adj_plot = cut_to_01(adj_plot)

        rec = net.forward(Y_0);
        rec = rec.detach().cpu().numpy()
        rec = np.squeeze(rec)
        rec_plot = cut_to_01(rec)
        X_0_plot = X_0.cpu().numpy()
        X_0_plot = np.squeeze(X_0_plot)
        
        err_im = np.abs(X_0_plot - rec_plot);
        diff = np.amax(err_im);
        th = 0.2;
        err_im[err_im > th] = 1;

        if diff > 0.22:
            print(f"{i:02d}/{nbr_of_images}: Diff: {diff:0.3f}")
            im = np.zeros([2*N+bd, 2*N+bd], 'uint8')
            im[:N,:N] = np.uint8(255*X_0_plot)
            im[:N,N+bd:] = np.uint8(255*rec_plot)
            im[N+bd:,:N] = np.uint8(255*adj_plot)
            im[N+bd:,N+bd:] = np.uint8(255*err_im)
            
            # Concatenated image
            pil_im = Image.fromarray(im);
            pil_im.save(join(image_location, f'mod_{model_nbr:03d}_{dir1}_im_nbr_{i:03d}.png'))
            
            # Splitted images
            pil_im_plot = Image.fromarray(np.uint8(255*X_0_plot));
            pil_im_rec = Image.fromarray(np.uint8(255*rec_plot));
            pil_im_plot.save(join(split_location, f'mod_{model_nbr:03d}_{dir1}_im_nbr_{i:03d}_orig.png'))
            pil_im_rec.save(join(split_location, f'mod_{model_nbr:03d}_{dir1}_im_nbr_{i:03d}_rec.png'))

#rec_plot = torch.sqrt(rec.pow(2).sum(-3))[0,:,:].detach().cpu().numpy()
#
##plt.figure(); 
##plt.subplot(221); plt.matshow(np.squeeze(sample1), fignum=False, cmap='gray'); plt.title('x');  plt.axis('off')
##plt.subplot(222); plt.matshow(adj_plot, fignum=False, cmap='gray'); plt.title('Adjoint');  plt.axis('off')
##plt.subplot(223); plt.matshow(rec_plot, fignum=False, cmap='gray'); plt.title('Network reconstrution'); plt.axis('off')
##plt.subplot(224); plt.matshow(rec_tv_plot, fignum=False, cmap='gray'); plt.title('TV'); plt.axis('off')
##plt.show()
#
#dest = 'plots'
#
#def _cut_to_01(im):
#    im = np.squeeze(im)
#    print(im.shape)
#    idx_neg = im < 0
#    idx_pos = im > 1
#    print(idx_neg.shape)
#    im[idx_neg] = 0
#    im[idx_pos] = 1
#    return im
#
#
#pil_im_true = Image.fromarray(np.uint8(255*_cut_to_01(sample1.numpy())));
#pil_im_adj = Image.fromarray(np.uint8(255*_cut_to_01(adj_plot)));
#pil_im_net_rec = Image.fromarray(np.uint8(255*_cut_to_01(rec_plot)));
#pil_im_tv_rec = Image.fromarray(np.uint8(255*_cut_to_01(rec_tv_plot)));
#
#pil_im_true.save(join(dest, 'im_true.png'))
#pil_im_adj.save(join(dest, 'im_adj.png'))
#pil_im_net_rec.save(join(dest, 'im_net_rec.png'))
#pil_im_tv_rec.save(join(dest, 'im_tv_rec.png'))



