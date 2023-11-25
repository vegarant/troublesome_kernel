import torch
import scipy.io
from os.path import join

nbr_files = 25000
nbr_files = 1000
type_of_data = 'val' # 'train' 

parent_dir = "/mn/kadingir/vegardantun_000000/nobackup/ellipses"
src_pt_files = join(parent_dir, 'raw_data', type_of_data)
dest_mat_files = join(parent_dir, 'raw_data_mat', type_of_data)

for i in range(nbr_files):
    fname = join(src_pt_files, f'sample_{i}.pt')
    im = torch.load(fname)
    im = im.double().numpy()
    fname_out = join(dest_mat_files, f'sample_{i:05d}.mat');
    scipy.io.savemat(fname_out, {'im': im});




