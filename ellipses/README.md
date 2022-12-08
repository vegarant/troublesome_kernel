
This directory contains the script used to test the stability and accuracy of NNs in Figure 3. The name convention of some of the files follows that from the [Genzel, Macdonald, & März](https://github.com/jmaces/robust-nets).

To train the NNs, run the following scripts:

- Step 1: Run the script `script_train_fourier_unet_jitter_v3.py`.
- Step 2: Run the scripts `script_train_fourier_unet_it_jit-nojit_pre.py`. `script_train_fourier_unet_it_no_jitter.py` and `script_train_fourier_unet_it_jitter_v4.py`.
- Step 3: Run the scripts `script_train_fourier_unet_it_jit-nojit.py`.

All the code has been run using PyTorch version 1.9.0.

