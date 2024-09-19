# =========================================================================== #
# author:  Martin KUKRÁL                                                      #
# date:    September 19, 2024                                                 #
# licence: CC BY-NC 4.0                                                       #
# purpose: 1) define the CAE class                                            #
#          2) neural network training setup                                   #
#          3) training loop (with reordered channels)                         #
# =========================================================================== #
from tqdm import tqdm   # 4.66.1
import numpy as np      # 1.25.2
import torch            # 2.1.1+cu121
import torch.nn as nn
import time
import os





# set the device to train on (CPU or GPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# --- 1) Define the CAE class -------------------------------------------------
class CAE(nn.Module):
    def __init__(self):
        super(CAE, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #set the device
        # the encoder architecture:
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Conv2d(4, 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32), 
            nn.ReLU()
        )     
        # linear layers around the bottleneck:
        self.fcE = nn.Linear(64000, 1000)
        self.fcD = nn.Linear(1000, 64000)
        self.relu = nn.ReLU()
        # the decoder architecture:
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.ConvTranspose2d(8, 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.ConvTranspose2d(4, 1, kernel_size=4, stride=2, padding=1)
        )
    
    def encode(self, x):
        ########################################################################
        # x - input EEG chunk                                                  #
        ########################################################################
        h = self.encoder(x)
        h = self.fcE(h.view(-1, 64000))
        return h
    
    def decode(self, h):
        ########################################################################
        # h - latent-space representation                                      #
        ########################################################################
        x_ = self.fcD(h)
        x_ = self.relu(x_)
        x_ = self.decoder(x_.view(-1, 32, 16, 125))
        return x_
    
    def forward(self, x):
        ########################################################################
        # x - input EEG chunk                                                  #
        ########################################################################
        h = self.encode(x)
        x_ = self.decode(h)
        return x_, h
    









if __name__ == "__main__":
    # --- 2) Neural network training setup ------------------------------------
    # initialize model:
    model = CAE()
    model = model.to(model.device) #if available move to GPU
    # initialize optimizer and loss function:
    optim = torch.optim.NAdam(model.parameters(), lr=0.001)
    lf = torch.nn.L1Loss()
    # prepare tensor datasets:
    trainset = os.listdir("./data/train/") # training dataset, 20 000 chunks
    trainset.remove("tensors_40.pt")
    validset = torch.load("./data/train/tensors_40.pt").float() # validation dataset, 500 chunks
    testset = os.listdir("./data/test/") # testing dataset, 5 000 chunks

    # --- 3) Training loop (reordered channels) -------------------------------
    start = time.time()
    order = np.load("./outputs/reorder.npy") #the new order calculated using UPGMA
    # training data:
    losses_epoch = []
    losses_epochV = []
    for e in range(20):
        model.train() #training mode
        loss_tensor = []
        for tensors in tqdm(trainset):
            dataset = torch.load(f"./data/train/{tensors}").float()
            dataset = torch.from_numpy(np.cbrt(dataset.numpy()))
            loader = torch.utils.data.DataLoader(dataset[:, order, :], batch_size=10, shuffle=True)
            loss_batch = []
            for data in loader:
                data = data.unsqueeze(1).to(model.device)
                optim.zero_grad()
                out, _ = model(data)
                loss = lf(out, data)
                loss.backward()
                loss_batch.append(loss.item())
                optim.step()
            loss_tensor.append(np.mean(loss_batch))
        losses_epoch.append(loss_tensor)
        # validation data:
        model.eval() #evaluation mode
        loss_batchV = []
        loader = torch.utils.data.DataLoader(torch.from_numpy(np.cbrt(validset.numpy()))[:, order, :], batch_size=10, shuffle=True)
        with torch.no_grad():   
            for data in loader:
                data = data.unsqueeze(1).to(model.device)
                out, _ = model(data)
                loss = lf(out, data)
                loss_batchV.append(loss.item())
        losses_epochV.append(np.mean(loss_batchV))
        print(f"=== [{e+1}] ========================================")
        print(f"TRAIN:   {np.mean(loss_tensor)}")
        print(f"VALID:   {np.mean(loss_batchV)}\n")
        # save losses for visualizations:
        np.save("./outputs/trainlossesRE.npy", np.array(losses_epoch))
        np.save("./outputs/validlossesRE.npy", np.array(losses_epochV))
    # testing data:
    model.eval() #evaluation mode
    loss_tensor = []
    with torch.no_grad():
        for tensors in tqdm(testset):
            dataset = torch.load(f"./data/test/{tensors}").float()
            dataset = torch.from_numpy(np.cbrt(dataset.numpy()))
            loader = torch.utils.data.DataLoader(dataset[:, order, :], batch_size=10, shuffle=True)
            loss_batch = []
            for data in loader:
                data = data.unsqueeze(1).to(model.device)
                out, _ = model(data)
                loss = lf(out, data)
                loss_batch.append(loss.item())
            loss_tensor.append(np.mean(loss_batch))
    print("==================================================")
    print(f"TEST:   {np.mean(loss_tensor)}")
    print("==================================================")
    print(f"TRAINING TIME: {time.time() - start} s")
    np.save("./outputs/testlossesRE.npy", np.array(loss_tensor))
    torch.save(model.state_dict(), "./outputs/paramsRE.pt")