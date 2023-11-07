clear('all'); close('all');

N = 256;
im_nbr = 5;

src_data = sprintf('/mn/kadingir/vegardantun_000000/nobackup/CT_images/res_%d_png/im_nbr_%03d.png', N, im_nbr);
src_mat = fullfile('/mn/sarpanitu/ansatte-u4/vegarant/cilib_data_final/radon_matrices', sprintf('radonMatrix2N%d_ang50.mat', N));
src_hallu = sprintf('hallu_siam_%d_im_%03d.png', N, im_nbr);

load(src_mat); % A
im = double(imread(src_data))/256;
hallu = double(imread(src_hallu))/256;
hallu = hallu/4;
[m, ~] = size(A);
N2 = N*N;

%l1 = round(0.25*N);
%h1 = round(0.75*N);
%hallu = zeros(N,N);
%hallu(l1, l1:h1) = 0.5;
y_h = A*hallu(:);
y_im = A*im(:);

I = speye(N2);
tic
x_perp = (A'*A + 0.01*I) \ (A'*y_h);
x_perp = reshape(x_perp, N,N);
toc

hallu_null = hallu-x_perp;
nbr_angles = 50
theta = linspace(0, 180*(1-1/nbr_angles), nbr_angles);
FBP = @(y) iradon(reshape(y, m/nbr_angles, nbr_angles), theta);

fig = figure('visible', 'off');
subplot(231);
imagesc(im); colormap('gray');
title('Orignal image')
axis('equal')
colorbar();

subplot(232);
imagesc(hallu); colormap('gray');
title('Detail not in kernel')
axis('equal')
colorbar();

subplot(233);
imagesc(FBP(y_im)); colormap('gray');
title('FBP(Ax)')
axis('equal')

subplot(234);
imagesc(im+hallu_null); colormap('gray');
title('Image + detail in kernel')
axis('equal')
colorbar();

subplot(235);
imagesc(hallu_null); colormap('gray');
title('Detail in in kernel')
axis('equal')
colorbar();

subplot(236);
imagesc(FBP(y_im+A*hallu_null(:))); colormap('gray');
title('FBP(A(x+x_det))')
axis('equal')

saveas(fig, fullfile('plots',sprintf('exp1_hallu_N_%d.png', N)));

norm(A*hallu(:))
norm(A*hallu_null(:))



