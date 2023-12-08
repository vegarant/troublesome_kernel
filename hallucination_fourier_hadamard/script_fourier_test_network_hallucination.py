import os
import glob
import re

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
    Fourier,
    LearnableInverterFourier,
    RadialMaskFunc,
    rotate_real,
)


# ------ setup ----------
device = torch.device("cuda:0")
torch.cuda.set_device(0)
DATA_PATH =  '/mn/kadingir/vegardantun_000000/nobackup/pytorch_datasets/fastMRI_complex'

model_nbr = 9
epoch_nbr = -1
train_phase = 2
nbr_of_images = 20
dest_plots = 'plots'

config_file = join(config.RESULTS_PATH, f'model_{model_nbr:03d}', "hyperparameters.yml")
copyfile(config_file, join(dest_plots, f'hyp_mod_{model_nbr:03d}.yml'))

with open(config_file) as ymlfile:
    cgf = yaml.load(ymlfile)#, Loader=yaml.SafeLoader)

N = cgf['DATA']['N']
fname_patt = cgf['DATA']['fname_patt']
srate = cgf['DATA']['srate']
subnet_name = cgf['DATA']['subnet']
learnableInv = cgf['DATA']['learnableInv']

print(f'PID: {os.getpid()}')
print(f'Model number: {model_nbr}')
print(f'N = {N}')
print(f'srate = {srate}')
print(f'SUBNET: {subnet_name}, {type(subnet_name)}')

if subnet_name == "<class 'networks.UNet'>":
    subnet = UNet
elif subnet_name == "<class 'networks.Tiramisu'>":
    subnet = Tiramisu
else:
    print('Error: subnet {subnet_name} is unknown')

n = (N, N)
mask = read_sampling_pattern_from_file(fname_patt);

# ----- operators -----
OpA = Fourier(mask)

subnet_params = cgf['SUBNET'].copy()
it_net_params = cgf['IT_NET'].copy()
it_net_params['operator'] = OpA
inverter = LearnableInverterFourier(n, mask, learnable=learnableInv)
#if it_net_params['inverter'].lower() == 'CheapNonLearnableInverter'.lower():
#   it_net_params['inverter'] = CheapNonLearnableInverter(OpA)
#else:
#    print('Error: Not implemented yet')
it_net_params['inverter'] = inverter
# loss
mseloss = torch.nn.MSELoss(reduction="sum")

def _complexloss(reference, prediction):
    loss = mseloss(
        rotate_real(reference)[:, 0:1, ...],
        rotate_real(prediction)[:, 0:1, ...],
    )
    return loss

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
    epoch_name = f'model_weights_epoch{epoch_nbr}.pt'
else: 
    epoch_name = "model_weights.pt"

net = _load_net(
    join(net_location, epoch_name),
    subnet,
    subnet_params,
    it_net_params
    )

dirs = ['train', 'val']
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
        fname = f'sample_{i:05d}.pt'
        if os.path.exists(join(DATA_PATH, dir1, fname)):
            sample = torch.load(join(DATA_PATH, dir1, fname))
            sample1 = sample.unsqueeze(0);
            X_0 = sample1.to(device)

            it_init = 1
            #print('hei: ', (X_0.ndim-1)*(1,))
            #X_0 = X_0.repeat(it_init, *((X_0.ndim - 1) * (1,)))
            #print('X_0.shape: ', X_0.shape)
            Y_0 = OpA(X_0)

            adj = OpA.adj(Y_0)
            adj_plot = adj.detach().cpu().numpy() 
            adj_plot = np.squeeze(adj_plot)
            adj_plot = np.sqrt(adj_plot[0,:,:]**2 + adj_plot[1,:,:]**2)
            adj_plot = cut_to_01(adj_plot)

            rec = net.forward(Y_0)
            rec = rec.detach().cpu().numpy()
            rec = np.squeeze(rec)
            rec_plot = np.sqrt(rec[0,:,:]**2 + rec[1,:,:]**2)
            rec_plot = cut_to_01(rec_plot)
            X_0 = torch.sqrt(X_0.pow(2).sum(-3))
            X_0_plot = X_0.cpu().numpy()
            X_0_plot = np.squeeze(X_0_plot)

            diff = np.amax(np.abs(X_0_plot - rec_plot))

            im = np.zeros([2*N+bd, 2*N+bd], 'uint8')
            im[:N,:N] = np.uint8(255*X_0_plot)
            im[:N,N+bd:] = np.uint8(255*rec_plot)
            im[N+bd:,:N] = np.uint8(255*adj_plot)
            im[N+bd:,N+bd:] = np.uint8(255*abs(X_0_plot - rec_plot))
            
            # Concatenated image
            pil_im = Image.fromarray(im);
            pil_im.save(join(image_location, f'mod_{model_nbr:03d}_{dir1}_im_nbr_{i:03d}.png'))
            
            # Splitted images
            pil_im_plot = Image.fromarray(np.uint8(255*X_0_plot));
            pil_im_rec = Image.fromarray(np.uint8(255*rec_plot));
            pil_im_plot.save(join(split_location, f'mod_{model_nbr:03d}_{dir1}_im_nbr_{i:03d}_orig.png'))
            pil_im_rec.save(join(split_location, f'mod_{model_nbr:03d}_{dir1}_im_nbr_{i:03d}_rec.png'))


dirs = ['train', 'test']
search_word = ['*halu.pt', '*.pt', '*mod.pt']
for s in range(len(dirs)):
    dir1 = dirs[s]
    sword = search_word[s]
    image_location = join(dest_plots, dir1)
    split_location = join(image_location, split_loc)
    

    if not os.path.isdir(image_location):
        os.mkdir(image_location)
    if not os.path.isdir(split_location):
        os.mkdir(split_location)


    files = glob.glob(join(DATA_PATH, dir1, sword))
    print(files)
    for fname in files:
        i = re.findall(r'\d+', fname)
        print(i)
        i = int(i[-1])
        if os.path.exists(join(DATA_PATH, dir1, fname)):
            sample = torch.load(join(DATA_PATH, dir1, fname))
            sample1 = sample.unsqueeze(0);
            X_0 = sample1.to(device)
    
            it_init = 1
            #print('hei: ', (X_0.ndim-1)*(1,))
            #X_0 = X_0.repeat(it_init, *((X_0.ndim - 1) * (1,)))
            #print('X_0.shape: ', X_0.shape)
            Y_0 = OpA(X_0)

            adj = OpA.adj(Y_0)
            adj = torch.sqrt(adj.pow(2).sum(-3))
            adj_plot = adj.detach().cpu().numpy() 
            adj_plot = np.squeeze(adj_plot)
            adj_plot = cut_to_01(adj_plot)

            rec = net.forward(Y_0)
            rec = rec.squeeze()
            print('rec.shape: ', rec.shape)
            #torch.save(rec, fname)   
            rec = torch.sqrt(rec.pow(2).sum(-3))
            rec = rec.detach().cpu().numpy()
            rec_plot = np.squeeze(rec)
            rec_plot = cut_to_01(rec_plot)
            X_0 = torch.sqrt(X_0.pow(2).sum(-3))
            X_0_plot = X_0.cpu().numpy()
            X_0_plot = cut_to_01(np.squeeze(X_0_plot))

            diff = np.amax(np.abs(X_0_plot - rec_plot))

            im = np.zeros([2*N+bd, 2*N+bd], 'uint8')
            im[:N,:N] = np.uint8(255*X_0_plot)
            im[:N,N+bd:] = np.uint8(255*rec_plot)
            im[N+bd:,:N] = np.uint8(255*adj_plot)
            im[N+bd:,N+bd:] = np.uint8(255*abs(X_0_plot - rec_plot))

            # Concatenated image
            pil_im = Image.fromarray(im);
            print('i: ', i)
            pil_im_plot = Image.fromarray(np.uint8(255*X_0_plot));
            pil_im_rec = Image.fromarray(np.uint8(255*rec_plot));
            pil_im_adj = Image.fromarray(np.uint8(255*adj_plot));
            
            if "halu" in fname: 
                pil_im.save(join(image_location, f'mod_{model_nbr:03d}_{dir1}_im_nbr_{i:03d}_halu.png'))
                # Splitted images
                pil_im_plot.save(join(split_location, f'mod_{model_nbr:03d}_{dir1}_im_nbr_{i:03d}_halu_orig.png'))
                pil_im_rec.save(join(split_location, f'mod_{model_nbr:03d}_{dir1}_im_nbr_{i:03d}_halu_rec.png'))
                pil_im_adj.save(join(split_location, f'mod_{model_nbr:03d}_{dir1}_im_nbr_{i:03d}_halu_adj.png'))
            elif "mod" in fname: 
                pil_im.save(join(image_location, f'mod_{model_nbr:03d}_{dir1}_im_nbr_{i:03d}_mod.png'))
                # Splitted images
                pil_im_plot.save(join(split_location, f'mod_{model_nbr:03d}_{dir1}_im_nbr_{i:03d}_mod_orig.png'))
                pil_im_rec.save(join(split_location, f'mod_{model_nbr:03d}_{dir1}_im_nbr_{i:03d}_mod_rec.png'))
                pil_im_adj.save(join(split_location, f'mod_{model_nbr:03d}_{dir1}_im_nbr_{i:03d}_mod_adj.png'))
            else:
                pil_im.save(join(image_location, f'mod_{model_nbr:03d}_{dir1}_im_nbr_{i:03d}.png'))
                # Splitted images
                pil_im_plot.save(join(split_location, f'mod_{model_nbr:03d}_{dir1}_im_nbr_{i:03d}_orig.png'))
                pil_im_rec.save(join(split_location, f'mod_{model_nbr:03d}_{dir1}_im_nbr_{i:03d}_rec.png'))
                pil_im_adj.save(join(split_location, f'mod_{model_nbr:03d}_{dir1}_im_nbr_{i:03d}_adj.png'))


