import h5py
import numpy as np
#import matplotlib.pyplot as plt
from PIL import Image
import glob
import os
from os.path import join

def remove_h5_ending(fname_core):
    if fname_core[-3:] == '.h5':
        fname_core = fname_core[:-3]
    return fname_core

def cut_to_interval(im, vmin, vmax):
    im1 = im.copy()
    idx_min = im1 < vmin
    idx_max = im1 > vmax
    im1[idx_min] = vmin
    im1[idx_max] = vmax
    return im1

def scale_to_01(im):
    im1 = im.copy()
    vmax = np.amax(im1)
    vmin = np.amin(im1)
    im1 -= vmin
    im1 /= (vmax-vmin)
    return im1

def create_comparison(rss, rec, fname_core, dest, acc, i, lim_inf, lim_l2, bd=5, splits='splits'):
    """
    lim (float): In the interval (0,1)
    """
    if not os.path.isdir(dest):
        os.mkdir(dest)
    if not os.path.isdir(join(dest,splits)):
        os.mkdir(join(dest, splits))

    fname_core = remove_h5_ending(fname_core)

    vmax = np.amax(rss)
    vmin = np.amin(rss)
    
    nrss = scale_to_01(rss)
    nrec = scale_to_01(cut_to_interval(rec, vmin, vmax))
    
    inf_norm = np.amax(np.abs(nrss-nrec))
    l2_norm = np.linalg.norm(nrss-nrec, ord='fro')
    if inf_norm > lim_inf and l2_norm < lim_l2:

        N1 = nrec.shape[0]
        N2 = nrec.shape[1]
        I = np.zeros([N1, 3*N2+2*bd], 'uint8')
        I[:,:] = np.uint8(255);
        I[:, :N2] = np.uint8(255*nrss)
        I[:, N2+bd:2*N2+bd] = np.uint8(255*nrec)
        I[:, 2*N2+2*bd:] = np.uint8(255*abs(nrec-nrss))
        
        im = Image.fromarray(I)
        tmp_name = join(dest,fname_core+f'_acc_{acc}_im_{i:02d}.png')
        print('Saving: ', tmp_name, ', norm inf: ', inf_norm, 'norm l2: ', l2_norm)
        im.save(tmp_name)
        
        im_true = Image.fromarray(np.uint8(255*nrss))    
        im_rec  = Image.fromarray(np.uint8(255*nrec))    
        im_true.save(join(dest, splits, fname_core + f'acc_{acc}_im_{i:02d}_true.png'))
        im_rec.save(join(dest, splits, fname_core + f'acc_{acc}_im_{i:02d}_rec.png'))

#fname = 'file_brain_AXT1_201_6002721.h5'
#fname = 'file_brain_AXT2_203_2030264.h5'
dest = 'plots'
acc = 8
lim_inf = 0.4
lim_l2 = 2000
src_rec = f'/mn/kadingir/vegardantun_000000/nobackup/fastMRI/multicoil_val_rec/rec_acc_{acc}'
src_data = '/mn/kadingir/vegardantun_000000/nobackup/fastMRI/multicoil_val_small'
for fname_full in glob.glob(join(src_data, '*.h5')):
    fname_full = join(src_data, fname_full)
    _, fname_core = os.path.split(fname_full)
    with h5py.File(join(src_rec, fname_core), 'r') as hf:
        rec_batch = np.array(hf['reconstruction'])
    with h5py.File(join(src_data, fname_core), 'r') as hf:
        data = np.array(hf['reconstruction_rss'])
    for i in range(rec_batch.shape[0]):
        create_comparison(data[i], rec_batch[i], fname_core, dest, acc, i, lim_inf, lim_l2)



