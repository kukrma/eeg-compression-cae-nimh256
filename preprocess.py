# =========================================================================== #
# author:  Martin KUKRÁL                                                      #
# date:    September 19, 2024                                                 #
# licence: CC BY-NC 4.0                                                       #
# purpose: 1) split the data                                                  #
#          2) load and shuffle the EEG chunks                                 #
#          3) make training datasets (+ validation dataset)                   #
#          4) make testing datasets                                           #
# =========================================================================== #
import mne              # 1.6.1
from tqdm import tqdm   # 4.66.1
import torch            # 2.1.1+cu121
import numpy as np      # 1.25.2
np.random.seed(0)
import os





# --- 1) Split the EEG data into chunks ---------------------------------------
# prepare empty folder for the chunks:
if not os.path.exists("./data/npy_chunks"): os.mkdir("./data/npy_chunks")
# pass through all raw data:
for subj in os.listdir("./raw"):
    print(f"=== {subj} ============================================================")
    raw = mne.io.read_raw_fieldtrip(f"./raw/{subj}", info=None, data_name="dataRaw") #load raw data
    raw = raw.drop_channels(("VREF", "vEOG", "hEOG", "SWL")).crop(5, raw.times[-5000]) #remove MISC channels and 0s
    data = raw._data #convert to NumPy array
    print("Splitting data into chunks...")
    dim = data.shape[1] #number of samples
    fits = dim // 2000 #number of resulting chunks
    data = data[:, :fits*2000] #clip the data to size
    chunks = np.array(np.split(data, data.shape[1]/2000, axis=1)) #reshape to [chunks, channels, samples]
    print("Saving chunks...")
    # pass through all chunks:
    for i, chunk in enumerate(chunks):
        np.save(f"./data/npy_chunks/{subj.split('.')[0]}_{i}.npy", chunk) #save chunk
    print("\n\n\n")

# --- 2) Load and shuffle chunks ----------------------------------------------
names = np.array(os.listdir("./data/npy_chunks")) #load list of chunk names
np.random.shuffle(names) #shuffle it randomly

# --- 3) Training datasets ----------------------------------------------------
# (the 41st dataset will be used as a validation dataset)
print("=== TRAIN SET PROCESSING =======================================================")
names_train = names[:20500] #names of chunks for training
divisible = np.array(np.split(names_train, 41)) #make as much equally sized splits
print("Making tensor datasets...")
# pass throught all equally sized splits:
for i, split in enumerate(divisible):
    tensors = torch.from_numpy(np.array([np.load(f"./data/npy_chunks/{name}") for name in tqdm(split)])) #tensor sized [500, channels, samples]
    torch.save(tensors, f"./data/train/tensors_{i}.pt") #save tensor

# --- 4) Testing datasets -----------------------------------------------------
print("=== TEST SET PROCESSING ========================================================")
names_test = names[20500:25500] #names of chunks for testing
divisible = np.array(np.split(names_test, 10)) #make splits
print("Making tensor datasets...")
# pass throught all splits:
for i, split in enumerate(divisible):
    tensors = torch.from_numpy(np.array([np.load(f"./data/npy_chunks/{name}") for name in tqdm(split)])) #tensor sized [500, channels, samples]
    torch.save(tensors, f"./data/test/tensors_{i}.pt") #save tensor