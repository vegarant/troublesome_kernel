N = 256;

src_data = sprintf('/mn/kadingir/vegardantun_000000/nobackup/CT_images/res_%d_png', N);
src_dest = '/mn/kadingir/vegardantun_000000/nobackup/CT_images/mat_files'; 

src_mat = '/mn/sarpanitu/ansatte-u4/vegarant/cilib_data_final/radon_matrices';
load(fullfile(src_mat, sprintf('radonMatrix2N%d_ang50.mat', N))); % A
[m, ~] = size(A);

nbr_angles = 50;
theta = linspace(0, 180*(1-1/nbr_angles), nbr_angles);
sinogram = @(y) reshape(y, m/nbr_angles, nbr_angles);
FBP = @(y) iradon(sinogram(y), theta, 'linear', 'Ram-Lak', 1, N);

for i = 1:95

    if (i ~= 5)
        fname = fullfile(src_data, sprintf('im_nbr_%03d.png', i));
        im = double(imread(fname)) / 256;
        y = A*im(:);
        im_FBP = FBP(y);
        fname_out = fullfile(src_dest, 'train', sprintf('sample_%05d.mat',i));
        save(fname_out, 'im', 'im_FBP');
    end

end

for i = 96:100

    fname = fullfile(src_data, sprintf('im_nbr_%03d.png', i));
    im = double(imread(fname)) / 256;
    y = A*im(:);
    im_FBP = FBP(y);
    fname_out = fullfile(src_dest, 'val', sprintf('sample_%05d.mat',i));
    save(fname_out, 'im', 'im_FBP');

end

