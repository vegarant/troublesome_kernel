import torch
import numpy as np
from PIL import Image
import os
from os.path import join
from extra_utils import (
    read_sampling_pattern_from_file, 
    nullspace_proj,
    cut_to_01
)
from operators import Fourier
import matplotlib.pyplot as plt

DATA_PATH =  '/mn/kadingir/vegardantun_000000/nobackup/pytorch_datasets/fastMRI_complex/train'
src_hal = '/mn/kadingir/vegardantun_000000/nobackup/storage_all/storage3/hallucination' 
src_patt = './samp_patt'
srate = 0.2
N = 256;
nbr_levels = 50
a = 1.75
r0 = 0

bd = 5;
modality = 'fourier'

im_nbrs = [4]
for im_nbr in im_nbrs:
    print('im_nbr: ', im_nbr)
    im = np.array(Image.open(join(src_hal, f'halu_{modality}_{im_nbr:05d}.png'))) / 255
    Z = np.zeros([2,N,N])
    Z[0,:,:] = im
    Z = torch.tensor(Z, dtype=torch.float)
    
    fname_patt = join(src_patt, f'spf2_cgauss_N_{N}_srate_{int(100*srate):02d}_r_{nbr_levels}_r0_{r0}_a_{int(100*a)}.png')
    mask = read_sampling_pattern_from_file(fname_patt);
    print(f'PID: {os.getpid()}')
    print('N: ', N)
    print('srate: ', srate)
    
    OpA = Fourier(mask)
    Z = nullspace_proj(Z, OpA);
    print('Z.max(): ', torch.sqrt(Z.pow(2).sum(-3)).max())
    print('Z.min(): ', torch.sqrt(Z.pow(2).sum(-3)).min())
    src_file = f'sample_{im_nbr:05d}_mod.png'
    X = Image.open(join(src_hal, src_file))
    X = np.array(X)/255
    if len(X.shape) == 3:
        X = X[:,:, 0];
    X_copy = X.copy()
    print('X.shape: ', X.shape)
    X_torch = torch.zeros([2,N,N], dtype=torch.float);
    X_torch[0,:,:] = torch.tensor(X)
    X_torch = X_torch + Z;
    torch.save(X_torch, join(DATA_PATH, src_file[:-4]+"_halu.pt"))
    # Move old file
    #os.rename(join(DATA_PATH, src_file), join(DATA_PATH, '..', src_file))
    
    #X_copy = torch.sqrt(X_copy.pow(2).sum(-3)).detach().numpy()
    X = torch.sqrt(X_torch.pow(2).sum(-3)).detach().numpy()
    out = np.zeros([N,2*N+bd], 'uint8')
    out[:, :N] = np.uint8(255*cut_to_01(X_copy))
    out[:, N+bd:] = np.uint8(255*cut_to_01(X))

    pil_im_orig = Image.fromarray(np.uint8(255*cut_to_01(X_copy)));
    pil_im_orig.save(join('plots', f'sample_{im_nbr:05d}_orig.png'))
    
    pil_im = Image.fromarray(out);
    pil_im.save(join('plots', f'sample_{im_nbr:05d}.png'))



