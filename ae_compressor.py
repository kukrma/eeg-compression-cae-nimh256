# =========================================================================== #
# author:  Martin KUKRÁL                                                      #
# date:    September 19, 2024                                                 #
# licence: CC BY-NC 4.0                                                       #
# purpose: 1) define the Compressior class                                    #
#          2) use it with the pretrained CAE to compress/decompress           #
# =========================================================================== #
import torch                      # 2.1.1+cu121
import numpy as np                # 1.25.2
import matplotlib.pyplot as plt   # 3.7.2
import seaborn as sns             # 0.13.0
from tqdm import tqdm             # 4.66.1
from ae_training import CAE
import pickle
import os
import time





# --- 1) The compressor class -------------------------------------------------
class Compressor():
    def __init__(self, model_params, reorder_list=None):
        ###########################################################################
        # model_params - path to model parameters                                 #
        # reorder_list - path to NPY file with new order of channels from UPGMA,  #
        #                if set to None (default), no reordering is performed     #
        ###########################################################################
        # prepare model:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #prepare device
        self.model = CAE()
        self.model.load_state_dict(torch.load(model_params)) #load parameters
        self.model.device = device
        self.model.to(device)
        self.model.eval()
        # set reordering status:
        self.reordered = False
        if reorder_list:
            self.reordered = True #set to determine conditions later
            self.order = np.load(reorder_list) #load the reordered channels
    
    def rle_encode(self, matrix):
        ###########################################################################
        # matrix - the NumPy array to be encoded by RLE                           #
        ###########################################################################
        flat_matrix = np.pad(matrix.numpy().astype(np.uint8).flatten(), pad_width=(1, 0)) #pad with 0
        changes = np.diff(flat_matrix, prepend=flat_matrix[0]) != 0 #determine the changes
        change_positions = np.where(changes)[0] - 1 #get indices corrected for padding
        return change_positions
    
    def rle_decode(self, encoded):
        ###########################################################################
        # encoded - decode the RLE-encoded sequence                               #
        ###########################################################################
        decoded = np.zeros(512000).astype(np.int8) #empty array
        # if length is even:
        if len(encoded) % 2 == 0:
            reshaped_array = encoded.reshape(-1, 2) #prepare pairs of indices (i.e. intervals of 1s)
            # set 1s for each interval:
            for pair in reshaped_array:
                decoded[pair[0]:pair[1]] = 1
        # if length is odd:
        else:
            decoded[encoded[-1]:] = 1 #treat as a sequence all the way to the end
            reshaped_array = encoded[:-1].reshape(-1, 2) #prepare pairs of indices (i.e. intervals of 1s)
            # set 1s for each interval:
            for pair in reshaped_array:
                decoded[pair[0]:pair[1]] = 1
        return decoded.reshape(256, 2000).astype(np.bool_)
    
    def compress_dataset(self, dataset_path, output_folder, threshold, plots=False):
        ###########################################################################
        # dataset_path  - path to tensor dataset to be compressed                 #
        # output_folder - where to save the folder containing compressed chunks   #
        # threshold     - maximum tolerable error in amplitude                    #
        # plots         - whether to plot comparisons between original and        #
        #                 restored (NO corrections, i.e. only CAE reconstruction) #
        #                 EEG chunks                                              #
        ###########################################################################
        print("=== COMPRESSING TENSOR DATASET ===============")
        start = time.time() #set initial timer
        # prepare for saving:
        filename = os.path.split(dataset_path)[-1][:-3]
        if not os.path.exists(f"{output_folder}/{filename}"): os.makedirs(f"{output_folder}/{filename}")
        # loop through chunks:
        for i, chunk in tqdm(enumerate(torch.load(dataset_path).float())):
            if self.reordered:
                chunk = torch.from_numpy(np.cbrt(chunk[self.order, :].numpy())).unsqueeze(0).unsqueeze(0)
            else:
                chunk = torch.from_numpy(np.cbrt(chunk.numpy())).unsqueeze(0).unsqueeze(0)
            # pass the chunk through the model:
            x, z = self.model(chunk.to(self.model.device))
            x = torch.pow(x, 3)
            chunk = torch.pow(chunk, 3)
            diff = chunk - x.cpu()
            # calculate bad indices and corresponding values:
            bad = torch.abs(diff.squeeze()) > threshold
            values = torch.masked_select(diff, bad)
            d = {"z": z, "idx": self.rle_encode(bad), "val": values}
            with open(os.path.join(f"{output_folder}/{filename}", f"{filename}_{i:05}.pkl"), "wb") as fp:
                pickle.dump(d, fp)
            # plots if desired:
            if plots:
                # comparison of original and reconstructed chunk:
                min = torch.min(torch.min(x), torch.min(chunk)).item()
                max = torch.max(torch.max(x), torch.max(chunk)).item()
                _, axs = plt.subplot_mosaic([["orig", "cbar"], ["reconst", "cbar"]], width_ratios=(0.95, 0.05))
                heat_orig = sns.heatmap(np.squeeze(chunk.numpy()), cmap="viridis", ax=axs["orig"], vmin=min, vmax=max, cbar=False)
                heat_orig.set_xticks([])
                heat_orig.set_yticks([])
                axs["orig"].set_xlabel("samples")
                axs["orig"].set_ylabel("channels")
                axs["orig"].set_title("ORIGINAL")
                heat_reconst = sns.heatmap(np.squeeze(x.detach().cpu().numpy()), cmap="viridis", ax=axs["reconst"],
                                           vmin=min, vmax=max, cbar_ax=axs["cbar"])
                heat_reconst.set_xticks([])
                heat_reconst.set_yticks([])
                axs["reconst"].set_xlabel("samples")
                axs["reconst"].set_ylabel("channels")
                axs["reconst"].set_title("RESTORED")
                axs["cbar"].set_title("μV")
                plt.tight_layout()
                plt.show()
                # absolute error plot:
                heat_err = sns.heatmap(np.squeeze(torch.abs(diff).detach().numpy()), cmap="Reds")
                heat_err.set_xticks([])
                heat_err.set_yticks([])
                plt.xlabel("samples")
                plt.ylabel("channels")
                plt.title("ABSOLUTE ERROR")
                plt.show()
        print(f"compression time: {time.time() - start}")
    
    def decompress_dataset(self, compressed_folder, output_folder, restore_dataset=True, float64=True):
        ###########################################################################
        # compressed folder - path to the folder with compressed chunks           #
        # output_folder     - where to save the decompressed file                 #
        # restore_dataset   - whether to restore the original tensor dataset or   #
        #                     or not (adds some computational time); if set to    #
        #                     False, each EEG chunk will be a separate file       #
        # float64           - whether to convert the decompressed file back to    #
        #                     float64, so that the size of decompressed files is  #
        #                     the same as before the compression (as CAE outputs  #
        #                     the chunks in float32)                              #
        ###########################################################################
        print("=== DECOMPRESSING TENSOR DATASET ===============")
        start = time.time() #initial time
        # load all files from the compressed folder:
        files = sorted(os.listdir(compressed_folder))
        orig_name, _ = files[0].split(".")
        orig_name = orig_name[:-6]
        for i, file in tqdm(enumerate(files)):
            d = pickle.load(open(f"{compressed_folder}/{file}", "rb"))
            decompressed = torch.pow(torch.squeeze(self.model.decode(d["z"])), 3)
            mask = self.rle_decode(d["idx"])
            decompressed[mask] = torch.add(decompressed[mask], d["val"].to(self.model.device))
            if float64:
                torch.save(decompressed.cpu().detach().to(torch.float64), f"{output_folder}/chunk_{i:05}.pt")
            else:
                torch.save(decompressed.cpu().detach(), f"{output_folder}/chunk_{i:05}.pt")
        print(f"decompression time (chunks): {time.time() - start} s")
        if restore_dataset:
            decompfiles = [fname for fname in os.listdir(output_folder) if "chunk" in fname]
            tensors = torch.stack([torch.load(f"{output_folder}/{name}") for name in decompfiles])
            torch.save(tensors, f"{output_folder}/{orig_name}.pt")
            for name in decompfiles:
                os.remove(f"{output_folder}/{name}")
        print(f"decompression time (dataset): {time.time() - start} s")
    
    def evaluate_dataset(self, original_path, restored_path, plots=False):
        ###########################################################################
        # original_path - path to the original tensor dataset                     #
        # restored_path - path to the restored tensor dataset                     #
        # plots         - whether to plot the original and restored EEG chunks    #
        #                 (with corrections)                                      #
        ###########################################################################
        original = torch.load(original_path)[:, self.order, :] if self.reordered else torch.load(original_path)
        restored = torch.load(restored_path)
        diff = original - restored
        prd = torch.sqrt(torch.sum(diff**2)/torch.sum(original**2))*100
        rmse = torch.sqrt(torch.sum(diff**2)/torch.numel(diff))
        print(f"PRD: {prd.item()}  |  RSME: {rmse.item()}")
        # plots:
        if plots:
            # comparison of original and reconstructed (with corrections) chunks
            for i in range(len(original)):
                min = torch.min(torch.min(original[i]), torch.min(restored[i])).item()
                max = torch.max(torch.max(original[i]), torch.max(restored[i])).item()
                _, axs = plt.subplot_mosaic([["orig", "cbar"], ["reconst", "cbar"]], width_ratios=(0.95, 0.05))
                heat_orig = sns.heatmap(np.squeeze(original[i].numpy()), cmap="viridis", ax=axs["orig"], vmin=min, vmax=max, cbar=False)
                axs["orig"].set_xlabel("samples")
                axs["orig"].set_ylabel("channels")
                axs["orig"].set_title("ORIGINAL")
                heat_orig.set_xticks([])
                heat_orig.set_yticks([])
                heat_reconst = sns.heatmap(np.squeeze(restored[i].detach().cpu().numpy()), cmap="viridis", ax=axs["reconst"], vmin=min, vmax=max, cbar_ax=axs["cbar"])
                axs["reconst"].set_xlabel("samples")
                axs["reconst"].set_ylabel("channels")
                axs["reconst"].set_title("RESTORED (WITH CORRECTIONS)")
                axs["cbar"].set_title("μV")
                heat_reconst.set_xticks([])
                heat_reconst.set_yticks([])
                plt.tight_layout()
                plt.show()
                # absolute error plot:
                heat_diff = sns.heatmap(np.squeeze(torch.abs(diff[i]).detach().numpy()), cmap="Reds")
                plt.xlabel("samples")
                plt.ylabel("channels")
                plt.title("ABSOLUTE ERROR")
                heat_diff.set_xticks([])
                heat_diff.set_yticks([])
                plt.show()










if __name__ == "__main__":
    # --- 2) Use the Compressor with the pre-trained CAE ----------------------
    compressor = Compressor("./outputs/params.pt") # the variant with the original order of channels
    #compressor = Compressor("./outputs/paramsRE.pt", reorder_list="./outputs/reorder.npy") #the variant with reordered channels

    # loop for compressing multiple datasets using multiple thresholds:
    '''
    for t in (0, 2, 4, 6, 8, 10): # thresholds
        for i in range(10): # numbers of tensor datasets
            print(f"DATASET {i}")
            compressor.compress_dataset(f"./data/float32/tensors32_{i}.pt", f"./outputs/compressed/{i}uV", t)
            compressor.decompress_dataset(f"./outputs/compressed/{i}uV/tensors32_{i}", "./outputs/decompressed", float64=False)
            compressor.evaluate_dataset(f"./data/float32/tensors32_{i}.pt", f"./outputs/decompressed/tensors32_{i}.pt")
    '''
    # compressing/decompressing only single tensor dataset:
    '''
    compressor.compress_dataset("./data/float32/tensors32_0.pt", "./outputs/compressed/TEMP", 3)
    compressor.decompress_dataset("./outputs/compressed/TEMP/tensors32_0", "./outputs/decompressed")
    compressor.evaluate_dataset("./data/float32/tensors32_0.pt", "./outputs/decompressed/tensors32_0.pt")
    '''