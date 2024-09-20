# =========================================================================== #
# author:  Martin KUKRÁL                                                      #
# date:    September 19, 2024                                                 #
# Python:  3.11.4                                                             #
# licence: CC BY-NC 4.0                                                       #
# purpose: 1) loss curve plot                                                 #
# =========================================================================== #
import numpy as np                # 1.25.2
import matplotlib.pyplot as plt   # 3.7.2





# --- 1) Loss curve plot ------------------------------------------------------
# load losses:
trainloss = np.load("./outputs/trainlosses.npy")
validloss = np.load("./outputs/validlosses.npy")
testloss = np.load("./outputs/testlosses.npy")
trainlossRE = np.load("./outputs/trainlossesRE.npy")
validlossRE = np.load("./outputs/validlossesRE.npy")
testlossRE = np.load("./outputs/testlossesRE.npy")
# calculate parameters to plot:
mean_train = np.mean(trainloss, axis=1)
std_train = np.std(trainloss, axis=1)
mean_test = np.mean(testloss)
std_test = np.std(testloss)
mean_trainRE = np.mean(trainlossRE, axis=1)
std_trainRE = np.std(trainlossRE, axis=1)
mean_testRE = np.mean(testlossRE)
std_testRE = np.std(testlossRE)
# make plot:
fig, axs = plt.subplot_mosaic([["orig"], ["reord"]])
axs["orig"].fill_between(x=range(1, 21), y1=mean_train+std_train, y2=mean_train-std_train, color="mediumseagreen", alpha=0.2)
axs["orig"].plot(range(1, 21), mean_train, color="mediumseagreen", linewidth=2, label="mean of training datasets (with ± STD)")
axs["orig"].scatter(20, mean_test, marker="_", s=500, color="darkorange", label="mean of testing datasets (with ± STD)")
axs["orig"].errorbar(20, mean_test, yerr=std_test, color="darkorange", capsize=5, alpha=0.5)
axs["orig"].plot(range(1, 21), validloss, color="cornflowerblue", linewidth=2, label="validation dataset")
axs["orig"].set_xlabel("epochs")
axs["orig"].set_ylabel("loss")
axs["orig"].set_xticks(range(1, 21)[1::2])
axs["orig"].grid(color="silver", linestyle="--", linewidth=0.5)
axs["orig"].set_title("ORIGINAL")
axs["reord"].fill_between(x=range(1, 21), y1=mean_trainRE+std_trainRE, y2=mean_trainRE-std_trainRE, color="mediumseagreen", alpha=0.2)
axs["reord"].plot(range(1, 21), mean_trainRE, color="mediumseagreen", linewidth=2, label="mean of training datasets (with ± STD)")
axs["reord"].scatter(20, mean_testRE, marker="_", s=500, color="darkorange", label="mean of testing datasets (with ± STD)")
axs["reord"].errorbar(20, mean_testRE, yerr=std_testRE, color="darkorange", capsize=5, alpha=0.5)
axs["reord"].plot(range(1, 21), validlossRE, color="cornflowerblue", linewidth=2, label="validation dataset")
axs["reord"].set_xlabel("epochs")
axs["reord"].set_ylabel("loss")
axs["reord"].set_xticks(range(1, 21)[1::2])
axs["reord"].grid(color="silver", linestyle="--", linewidth=0.5)
axs["reord"].set_title("REORDERED")
plt.legend(fontsize=10)
plt.show()