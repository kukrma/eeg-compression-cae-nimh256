# Near-lossless EEG Signal Compression Using a Convolutional Autoencoder: Case Study for 256-channel Binocular Rivalry Dataset
This GitHub repository contains the code used for the purposes of the article *Near-lossless EEG Signal Compression Using a Convolutional Autoencoder: Case Study for 256-channel Binocular Rivalry Dataset*, which is currently in works (once published, the necessary links to the article will be added).

The code is mirrored on Zenodo, where it is accompanied by one testing tensor dataset (explained in the article) for demonstration purposes, as well as the pre-trained parameters to the neural network. The original raw data is not made publicly available, but could be shared upon reasonable request.

[![CC BY-NC 4.0][cc-by-nc-shield]][cc-by-nc]

[![CC BY-NC 4.0][cc-by-nc-image]][cc-by-nc]

[cc-by-nc]: https://creativecommons.org/licenses/by-nc/4.0/
[cc-by-nc-image]: https://licensebuttons.net/l/by-nc/4.0/88x31.png
[cc-by-nc-shield]: https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg

## Description of Files
If we follow the flow of the article, the order in which the files are executed is:
1) `overview.py` – loading and extracting the EEG signals using MNE, computing basic characteristics, and visualisation of the signal;
2) `stats_compute.py` – computing statistics (correlation, autocorrelation) and saving the results to files;
3) `stats_visualize.py` – visualization of statistical analysis results, applying UPGMA and saving the array indices used for reordering the channels during testing;
4) `preprocess.py` – splitting REG signals into chunks and shuffling them, making tensor datasets for training and testing the neural network;
5) `ae_training.py` – definition of the `CAE` (convolutional autoencoder) class and training the network on the original order of channels;
6) `ae_trainingRE.py` – same as `ae_training.py`, but with the reordered channels using UPGMA;
7) `ae_visualize.py` – visualization of the loss curve;
8) `ae_compressor.py` – definition of the `Compressor` class, which is used to compress and decompress tensor datasets, as well as to evaluate the compressions using CR (compression ratio), PRD (percentage root mean square difference), and RMSE (root mean square error);
9) `results_calculations.py` – calculations and plots of results.

## Original Directory Tree
The code contains paths reflecting the file structure used by the author, which can be altered. For clarity, it was as follows (files irrelevant to the code were excluded):
```
└── pyPROJECT/
    ├── data/
    |   ├── float32/
    |   |   ├── tensors32_0.pt
    |   |   ├── tensors32_1.pt
    |   |   ⋮
    |   |   └── tensors32_9.pt
    |   ├── npy_chunks/
    |   |   ├── subj_01_0.npy
    |   |   ├── subj_01_1.npy
    |   |   ⋮
    |   |   └── subj_25_1274.npy
    |   ├── test/
    |   |   ├── tensors_0.pt
    |   |   ├── tensors_2.pt
    |   |   ⋮
    |   |   └── tensors_9.pt
    |   └── train/
    |   |   ├── tensors_0.pt
    |   |   ├── tensors_2.pt
    |   |   ⋮
    |   |   └── tensors_40.pt
    ├── outputs/
    |   ├── compressed/
    |   |   ├── 0uV/
    |   |   ├── 2uV/
    |   |   ├── 4uV/
    |   |   ├── 6uV/
    |   |   ├── 8uV/
    |   |   └── 10uV/
    |   ├── decompressed/
    |   ├── autocorrelations.npy
    |   ├── correlations.npy
    |   ├── params.pt
    |   ├── paramsRE.pt
    |   ├── reorder.npy
    |   ├── testlosses.npy
    |   ├── testlossesRE.npy
    |   ├── trainlosses.npy
    |   ├── trainlossesRE.npy
    |   ├── validlosses.npy
    |   └── validlossesRE.npy
    ├── raw/
    |   ├── subj_01.mat
    |   ├── subj_02.mat
    |   ⋮
    |   └── subj_25.mat
    ├── ae_compressor.py
    ├── ae_training.py
    ├── ae_trainingRE.py
    ├── ae_visualize.py
    ├── overview.py
    ├── preprocess.py
    ├── results_calculations.py
    ├── stats_compute.py
    └── stats_visualize.py
```
