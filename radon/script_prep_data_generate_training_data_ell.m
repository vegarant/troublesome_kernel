src_data = '/mn/kadingir/vegardantun_000000/nobackup/ellipses/raw_data_mat';
src_dest = '/mn/kadingir/vegardantun_000000/nobackup/ellipses/raw_data_radon';

N = 256;
src_mat = '/mn/sarpanitu/ansatte-u4/vegarant/cilib_data_final/radon_matrices';
load(fullfile(src_mat, sprintf('radonMatrix2N%d_ang50.mat', N))); % A
[m, ~] = size(A);

nbr_angles = 50;
theta = linspace(0, 180*(1-1/nbr_angles), nbr_angles);
sinogram = @(y) reshape(y, m/nbr_angles, nbr_angles);
FBP = @(y) iradon(sinogram(y), theta, 'linear', 'Ram-Lak', 1, N);

train_or_val = {'train', 'val'};
nbr_of_samples_in_dir = {25000, 1000};
%train_or_val = {'val'};
%nbr_of_samples_in_dir = {1000};

for k = 1:2
    data_type = train_or_val{k};
    nbr_of_samples = nbr_of_samples_in_dir{k}; 
    
    for i = 1:nbr_of_samples
        src_data_full = fullfile(src_data, data_type);
        fname = fullfile(src_data_full, sprintf('sample_%05d.mat', i-1));
        load(fname); % im
        y = A*im(:);
        im_sinog = sinogram(y);
        im_FBP = FBP(y);
        fname_out = fullfile(src_dest,data_type, sprintf('sample_%05d.mat',i-1));
        save(fname_out, 'im', 'im_sinog', 'im_FBP');
    end
end

