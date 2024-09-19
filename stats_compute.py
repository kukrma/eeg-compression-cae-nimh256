# =========================================================================== #
# author:  Martin KUKRÁL                                                      #
# date:    September 19, 2024                                                 #
# Python:  3.11.4                                                             #
# licence: CC BY-NC 4.0                                                       #
# purpose: 1) calculate statistics (correlation and autocorrelation)          #
#          2) save the results to files                                       #
# =========================================================================== #
import mne                     # 1.6.1
import statsmodels.api as sm   # 0.14.0
from tqdm import tqdm          # 4.66.1
import numpy as np             # 1.25.2
np.random.seed(0)
import os





# --- 1) Calculate statistics -------------------------------------------------
# prepare empty lists for results:
corr_all = []
acf_all = []
# pass through all raw subject data:
for subj in os.listdir("./raw"):
    print(f"=== {subj} ============================================================")
    raw = mne.io.read_raw_fieldtrip(f"./raw/{subj}", info=None, data_name="dataRaw") #load raw data
    raw = raw.drop_channels(("VREF", "vEOG", "hEOG", "SWL")).crop(5, raw.times[-5000]) #remove MISC channels and 0s
    data = raw._data #extract signal as NumPy array
    print("Processing channels...")
    acf = np.array([sm.tsa.stattools.acf(channel, nlags=3000) for channel in tqdm(data)]) #calculate ACF for each channel
    corr = np.corrcoef(data, rowvar=True) #calculate interchannel correlation
    # append to prepared empty lists:
    corr_all.append(corr)
    acf_all.append(acf)
    print("\n\n\n")
corr_all = np.array(corr_all) #convert to NumPy array

# --- 2) Save results as files ------------------------------------------------
# (calculating the statistics takes a lot of time, so saving the results is recommended)
np.save("./outputs/correlations.npy", corr_all)
np.save("./outputs/autocorrelations.npy", acf_all)