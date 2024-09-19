# =========================================================================== #
# author:  Martin KUKRÁL                                                      #
# date:    September 19, 2024                                                 #
# Python:  3.11.4                                                             #
# licence: CC BY-NC 4.0                                                       #
# purpose: 1) loading and extracting the EEG data using MNE                   #
#          2) calculating basic characteristics of the signal                 #
#          3) EEG signal visualization                                        #
# =========================================================================== #
import mne                        # 1.6.1
import numpy as np                # 1.25.2
import seaborn as sns             # 0.13.0
import matplotlib                 # 3.7.2
import matplotlib.pyplot as plt
matplotlib.use("Qt5Agg") #(for MNE visualization)





# --- 1) Loading data as mne.io.Raw -------------------------------------------
# load raw data using MNE:
raw = mne.io.read_raw_fieldtrip("./raw/subj_01.mat", info=None, data_name="dataRaw")
# remove MISC channels and artificial 0s from the signal:
raw = raw.drop_channels(("VREF", "vEOG", "hEOG", "SWL")).crop(5, raw.times[-5000])
# extract only the EEG signal as NumPy array:
data = raw._data

# --- 2) Characteristics ------------------------------------------------------
print(f"MEAN: {np.mean(data)}")
print(f"MAX: {np.max(data)}")
print(f"MIN: {np.min(data)}")

# --- 3) EEG signal visualization ---------------------------------------------
'''
### REMOVED FROM THE FINAL VERSION OF THE ARTICLE ###
# interactive EEG plot:
raw.plot(scalings="auto")
plt.show()
'''
# EEG signal as an image:
heat = sns.heatmap(data[:, 7000:10000], cmap="viridis", annot=False)
heat.set_xticks([500, 1000, 1500, 2000, 2500, 3000])
heat.set_xticklabels(["500", "1000", "1500", "2000", "2500", "3000"])
heat.set_xticklabels(heat.get_xticks(), size = 8)
heat.set_yticks([])
plt.xlabel("samples")
plt.ylabel("channels")
plt.show()