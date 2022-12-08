import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from matplotlib import rc
from piq import psnr, ssim
from data_management import IPDataset
from find_adversarial import err_measure_l2, grid_attack
from operators import to_magnitude, to_complex

# ----- load configuration -----
import config  # isort:skip
import config_robustness_fourier as cfg_rob  # isort:skip
from config_robustness_fourier import methods  # isort:skip

# ------ general setup ----------
device = cfg_rob.device
torch.manual_seed(1)

save_path = os.path.join(config.RESULTS_PATH, "attacks")
save_plot = True
# select samples
sample = 968

# dynamic range for plotting & similarity indices
v_min = 0.0
v_max = 0.9

err_measure = err_measure_l2

# select reconstruction methods
methods_include = ["UNet it jit", "UNet it jit mod", "UNet it no jit", "L1"]
methods = methods.loc[methods_include]

# select methods excluded from (re-)performing attacks
methods_no_calc = []


# select sample

single_im = torch.load(os.path.join(config.DATA_PATH, f'test/sample_{sample}_text.pt'))
single_im1 = single_im.unsqueeze(0);

X_0 = to_complex(single_im1.to(device)).unsqueeze(0)

it_init = 1
print((X_0.ndim-1)*(1,))
X_0 = X_0.repeat(it_init, *((X_0.ndim - 1) * (1,)))
print('X_0.shape: ', X_0.shape)
Y_0 = cfg_rob.OpA(X_0)

X_0_cpu = X_0.cpu()



# ----- plotting -----
def _implot(sub, im, vmin=v_min, vmax=v_max):
    if im.shape[-3] == 2:  # complex image
        image = sub.imshow(
            torch.sqrt(im.pow(2).sum(-3))[0,:,:].detach().cpu(),
            vmin=vmin,
            vmax=vmax,
        )
    else:  # real image
        image = sub.imshow(im[0, 0, :, :].detach().cpu(), vmin=vmin, vmax=vmax)

    image.set_cmap("gray")
    sub.set_xticks([])
    sub.set_yticks([])
    return image

# LaTeX typesetting
rc("font", **{"family": "serif", "serif": ["Palatino"]})
rc("text", usetex=True)

# perform reconstruction
for (idx, method) in methods.iterrows():
    if idx not in methods_no_calc:
        X_rec = method.reconstr(Y_0, 0)
        rec_err = err_measure(X_rec, X_0);
        print(f'{idx}: rel l2 error: {rec_err}');

        fig, ax = plt.subplots(clear=True, figsize=(2.5, 2.5), dpi=200)
        im = _implot(ax, X_rec)
        ax.text(
            252,
            256,
            "rel.~$\\ell_2$-err: {:.2f}\\%".format(
                rec_err * 100
            ),
            fontsize=10,
            color="white",
            horizontalalignment="right",
            verticalalignment="bottom",
        )
        axins = ax.inset_axes([0.55, 0.75, 0.4, 0.2])
        _implot(axins, X_rec)

        axins.set_xlim(185, 235)
        axins.set_ylim(170, 145)
        axins.set_xticks([])
        axins.set_yticks([])
        axins.spines["bottom"].set_color("#a1c9f4")
        axins.spines["top"].set_color("#a1c9f4")
        axins.spines["left"].set_color("#a1c9f4")
        axins.spines["right"].set_color("#a1c9f4")
        ax.indicate_inset_zoom(axins, edgecolor="#a1c9f4")

        if save_plot:
            fig.savefig(
                os.path.join(
                    save_path,
                    "fig_example_S{}_adv_".format(sample)
                    + method.info["name_save"]
                    + "_text.pdf"
                ),
                bbox_inches="tight",
                pad_inches=0,
            )

        # not saved
        fig.suptitle(
            method.info["name_disp"] + " for unseen detail"
        )

        # error plot
        fig, ax = plt.subplots(clear=True, figsize=(2.5, 2.5), dpi=200)
        im = _implot(
            ax,
            (to_magnitude(X_rec) - to_magnitude(X_0)).abs(),
            vmin=0.0,
            vmax=0.6,
        )

        if save_plot and False:
            fig.savefig(
                os.path.join(
                    save_path,
                    "fig_example_S{}_adv_err_".format(sample)
                    + method.info["name_save"]
                    + "_text.pdf"
                ),
                bbox_inches="tight",
                pad_inches=0,
            )



