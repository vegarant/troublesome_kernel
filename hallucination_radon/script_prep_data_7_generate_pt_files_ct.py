import torch
import scipy.io
from tqdm import tqdm
from os.path import join
import os
import glob

src_data = '/mn/kadingir/vegardantun_000000/nobackup/CT_images/mat_files';
src_dest = '/mn/kadingir/vegardantun_000000/nobackup/CT_images/pt_files';

data_type = ['train', 'val']

for k in range(2):
    os.chdir(join(src_data, data_type[k]))
    for file_name in glob.glob('*.mat'):
        data_dict = scipy.io.loadmat(file_name)
        im = torch.tensor(data_dict['im'], dtype=torch.float)
        im_FBP = torch.tensor(data_dict['im_FBP'], dtype=torch.float)
        
        new_data_dict = {'im': im, 'im_FBP': im_FBP}
        
        fname_core = file_name[:-4]
        fname_out = join(src_dest, data_type[k], fname_core+'.pt')
        torch.save(new_data_dict, fname_out)

