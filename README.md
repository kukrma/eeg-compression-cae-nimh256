# 256-channel EEG Signal Compression Using a Convolutional Autoencoder: Binocular Rivalry Use Case
This GitHub repository contains the code used for the purposes of the article *Near-lossless EEG Signal Compression Using a Convolutional Autoencoder: Case Study for 256-channel Binocular Rivalry Dataset*, which is currently in works (once published, the necessary links to the article will be added).

The code is mirrored on Zenodo, where it is accompanied by one testing tensor dataset (see the article for explanation) for demonstration purposes, as well as the pre-trained parameters to the neural network. The original raw dataset is not made publicly available, but could be shared upon reasonable request.

[![CC BY-NC 4.0][cc-by-nc-shield]][cc-by-nc]

[![CC BY-NC 4.0][cc-by-nc-image]][cc-by-nc]

[cc-by-nc]: https://creativecommons.org/licenses/by-nc/4.0/
[cc-by-nc-image]: https://licensebuttons.net/l/by-nc/4.0/88x31.png
[cc-by-nc-shield]: https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg

## How to Use
To use this code, you need to first install Python on your computer and optionally some IDE in which you can comfortably interact with the scripts. Specifically, I have used **Python version 3.11.4** and the following libraries:
| LIBRARY     | VERSION     |
| ----------- | -------     |
| matplotlib  | 3.7.2       |
| mne         | 1.6.1       |
| numpy       | 1.25.2      |
| seaborn     | 0.13.0      |
| scipy       | 1.11.2      |
| statsmodels | 0.14.0      |
| torch       | 2.1.1+cu121 |
| tqdm        | 4.66.1      |

With everything prepared, the code should be ready to use. Given that we have performed a case study, the code is not fully generalized, but tailored specifically to the 256-channel binocular rivalry dataset by NIMH. If you want to apply this method to other EEG datasets, you would have to do some adjustments, such changing the way in which the EEG signal is loaded and preprocessed (if it is in some other format), changing the neural network architecture (if the number of channels is different) etc. In the future, we would like to improve upon this code and make it more generally usable, which was out of scope of the original paper.

The raw dataset used in the article is not made publicly available, but one testing tensor dataset and the pre-trained network parameters are published on Zenodo, so the `ae_compressor.py` script should be usable for demonstration purposes by anyone, once you download the `tensors32_0.pt` and `params.pt` files from Zenodo (alternativelly, you can use the `paramsRE.pt` with `reorder.npy` for the UPGMA version). In the `ae_compressor.py` script, you then add paths to these files when initializing the `Compressor` class, which will then allow you to perform compression using `.compress_dataset()`, decompression with `.decompress_dataset()`, and the evaluation with PRD and RMSE using `.evaluate_dataset()` (all three methods are parametrized by paths to relevant files). To apply additional folder compression to the compressed state, you need to manually use the 7-Zip software (or any other, but this was used in the study).

The source code files contain many comments with descriptions of individual steps and parameters used by the defined classes.


## Description of Files
If we follow the flow of the article, the order in which the files are executed is:
1) `overview.py` – loading and extracting the EEG signals using MNE, computing basic characteristics, and visualisation of the signal;
2) `stats_compute.py` – computing statistics (correlation, autocorrelation) and saving the results to files;
3) `stats_visualize.py` – visualization of statistical analysis results, applying UPGMA and saving the array indices used for reordering the channels during testing;
4) `preprocess.py` – splitting EEG signals into chunks and shuffling them, making tensor datasets for training and testing the neural network;
5) `ae_training.py` – definition of the `CAE` (convolutional autoencoder) class and training the network on the original order of channels;
6) `ae_trainingRE.py` – same as `ae_training.py`, but with the reordered channels using UPGMA;
7) `ae_visualize.py` – visualization of the loss curve;
8) `ae_compressor.py` – definition of the `Compressor` class, which is used to compress and decompress tensor datasets, as well as to evaluate the compressions using PRD (percentage root mean square difference) and RMSE (root mean square error);
9) `results_calculations.py` – calculations and plots of results.

## Original Directory Tree
The code contains paths reflecting the file structure used by the author, which can be altered. For clarity, it was as follows (files and folders irrelevant to the code were excluded):
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
