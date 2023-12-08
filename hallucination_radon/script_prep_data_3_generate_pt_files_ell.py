import torch
import scipy.io
from tqdm import tqdm
from os.path import join

src_data = '/mn/kadingir/vegardantun_000000/nobackup/ellipses/raw_data_radon';
src_dest = '/mn/kadingir/vegardantun_000000/nobackup/ellipses/raw_data_radon_pt';

nbr_of_samples = [25000, 1000]
data_type = ['train', 'val']

for k in range(2):
    print(f'data_type[k]')
    for i in tqdm(range(nbr_of_samples[k])):
        fname = join(src_data, data_type[k], f'sample_{i:05d}.mat')
        data_dict = scipy.io.loadmat(fname)
        im = torch.tensor(data_dict['im'], dtype=torch.float)
        im_FBP = torch.tensor(data_dict['im_FBP'], dtype=torch.float)
        new_data_dict = {'im': im, 'im_FBP': im_FBP}
        fname_out = join(src_dest, data_type[k], f'sample_{i:05d}.pt')
        torch.save(new_data_dict, fname_out)




