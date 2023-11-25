# Setup

This directory contains the code for producing the U-Net that hallucinates with the Radon sampling operator. It trains the neural network (NN) in two stages. In the pretraining step, the NN is trained on the 25 000 ellipses images created via the script in the ellipses directory. Then the NN is fine-tuned for the task of CT image reconstruction, by training on 95 images from [https://www.kaggle.com/datasets/kmader/siim-medical-images/](https://www.kaggle.com/datasets/kmader/siim-medical-images/). 

In this experiment, the matrix $A$ is generated as a sparse matrix using the Radon sampling operator in MATLAB. The code for generating this matrix can be found here:
[https://github.com/vegarant/cilib/blob/master/src/misc/Generate_radon_matrix.m](https://github.com/vegarant/cilib/blob/master/src/misc/Generate_radon_matrix.m).

Since we are using a sampling operator implemented in MATLAB and training the NN in Python, the data require some preprocessing. This is done as follows.

1. Store the images as `.mat` files.
2. Read these `.mat` files in MATLAB, sample them with the Radon sampling operator, and store the resulting samples together with the images in a new `.mat` file. 
3. Convert the `.mat` to `.pt` files to ensure fast reading of the data in PyTorch.

### References for the CT-data.

* Albertina, B., Watson, M., Holback, C., Jarosz, R., Kirk, S., Lee, Y., … Lemmerman, J. (2016). Radiology Data from The Cancer Genome Atlas Lung Adenocarcinoma [TCGA-LUAD] collection. The Cancer Imaging Archive. http://doi.org/10.7937/K9/TCIA.2016.JGNIHEP5

* Clark K, Vendt B, Smith K, Freymann J, Kirby J, Koppel P, Moore S, Phillips S, Maffitt D, Pringle M, Tarbox L, Prior F. The Cancer Imaging Archive (TCIA): Maintaining and Operating a Public Information Repository, Journal of Digital Imaging, Volume 26, Number 6, December, 2013, pp 1045-1057. (paper)



