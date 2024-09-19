# =========================================================================== #
# author:  Martin KUKRÁL                                                      #
# date:    September 19, 2024                                                 #
# licence: CC BY-NC 4.0                                                       #
# purpose: 1) plot the changes in compressed size for one tensor dataset      #
#          2) plot the CRs, PRDs, and RMSEs                                   #
# =========================================================================== #
import matplotlib.pyplot as plt   # 3.7.2
import numpy as np                # 1.25.2
import torch                      # 2.1.1+cu121





# --- 1) Plot of changes in the compressed size -------------------------------
# values:
thresholds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
base = [829, 517, 324, 210, 141, 98.6, 70.9, 52.6, 40.1, 31.3]
bzip2 = [623, 376, 233, 150, 101, 71.4, 52, 38.6, 29.8, 23.6]
deflate = [602, 357, 222, 144, 97.5, 68.3, 49.3, 36, 27.8, 21.9]
lzma = [544, 313, 196, 127, 86.2, 60.7, 43.6, 31.7, 24.5, 19.4]
ppmd = [630, 379, 235, 150, 101, 70.9, 51, 37.7, 29, 22.9]
# plot:
plt.plot(thresholds, base, "o-", color="dodgerblue", label="our method only")
plt.plot(thresholds, bzip2, "o-", color="mediumpurple", label="+BZip2")
plt.plot(thresholds, deflate, "o-", color="sandybrown", label="+Deflate")
plt.plot(thresholds, lzma, "o-", color="indianred", label="+LZMA")
plt.plot(thresholds, ppmd, "o-", color="mediumseagreen", label="+PPMd")
plt.xticks(thresholds)
plt.xlabel("thresholds [μV]")
plt.ylabel("compressed size [MB]")
plt.grid(color="silver", linestyle="--", linewidth=0.5)
plt.legend()
plt.show()

# --- 2) Plot the CRs, PRDs, and RMSEs ----------------------------------------
# CR values:
uv2_cr = [3.118, 3.04, 3.05, 3.04, 3.04, 3.118, 2.985, 3.079, 3.079, 3.04]
uv4_cr = [7.685, 7.45, 7.625, 7.45, 7.45, 7.808, 7.124, 7.625, 7.508, 7.394]
uv6_cr = [16.079, 15.443, 16.132, 15.492, 15.322, 16.403, 14.395, 15.767, 15.468, 14.993]
uv8_cr = [30.789, 28.876, 31.282, 29.576, 29.134, 31.182, 26.522, 29.756, 29.048, 27.493]
uv10_cr = [50.309, 45.395, 51.368, 48.079, 47.379, 50.051, 41.709, 47.379, 46.256, 42.807]
# PRD values:
uv2_prd = [15.485, 8.537, 10.297, 14.869, 15.249, 14.129, 2.494, 2.048, 14.008, 2.818]
uv4_prd = [30.142, 16.774, 20.241, 29.214, 29.901, 27.495, 4.916, 4.007, 27.363, 5.514]
uv6_prd = [39.733, 22.139, 26.726, 38.629, 39.564, 36.108, 6.526, 5.282, 36.125, 7.285]
uv8_prd = [46.060, 25.641, 30.935, 44.839, 45.985, 41.704, 7.603, 6.117, 41.925, 8.458]
uv10_prd = [50.380, 28.019, 33.765, 49.098, 50.417, 45.490, 8.352, 6.685, 45.921, 9.265]
# RMSE values:
uv2_rmse = [0.8211, 0.8197, 0.8212, 0.8193, 0.8190, 0.8234, 0.8153, 0.8210, 0.8196, 0.8186]
uv4_rmse = [1.5983, 1.6106, 1.6143, 1.6097, 1.6060, 1.6024, 1.6067, 1.6060, 1.6009, 1.6021]
uv6_rmse = [2.1068, 2.1258, 2.1315, 2.1284, 2.1250, 2.1044, 2.1329, 2.1168, 2.1135, 2.1167]
uv8_rmse = [2.4423, 2.4620, 2.4672, 2.4706, 2.4699, 2.4305, 2.4852, 2.4516, 2.4528, 2.4572]
uv10_rmse = [2.6714, 2.6904, 2.6929, 2.7053, 2.7079, 2.6511, 2.7298, 2.6793, 2.6866, 2.6919]
# the reconstruction when threshold set to 0:
uv0_prd = [1.4629558791057207e-05, 1.3373530237004161e-05, 1.3644954378833063e-05, 1.4417684724321589e-05, 1.4369727978191804e-05,
           1.4137498510535806e-05, 1.3227439012553077e-05, 1.287861232412979e-05, 1.4244677004171535e-05, 1.3339760698727332e-05]
uv0_rmse = [7.757325874990784e-07, 1.2841037460020743e-06, 1.088244175662112e-06, 7.944097433210118e-07, 7.717999892520311e-07,
            8.239230169237999e-07, 4.3233744690951426e-06, 5.161302851774963e-06, 8.333926189152407e-07, 3.875605671055382e-06]
# calculate and print mean +- 95% CI (change the "uv0_rmse" to calculate other trials)
means = np.mean(uv0_rmse) # mean across subjects
conf = 1.96*(np.std(uv0_rmse)/np.sqrt(10)) # 95% confidence interval across subjects
print(means)
print(conf)
# stack and norm:
stacked_cr = np.stack((uv2_cr, uv4_cr, uv6_cr, uv8_cr, uv10_cr)).T
stacked_cr = stacked_cr/np.max(stacked_cr)
stacked_prd = np.stack((uv2_prd, uv4_prd, uv6_prd, uv8_prd, uv10_prd)).T
stacked_prd = stacked_prd/np.max(stacked_prd)
stacked_rmse = np.stack((uv2_rmse, uv4_rmse, uv6_rmse, uv8_rmse, uv10_rmse)).T
stacked_rmse = stacked_rmse/np.max(stacked_rmse)
# plot:
means_cr = np.mean(stacked_cr, axis=0)
means_prd = np.mean(stacked_prd, axis=0)
means_rmse = np.mean(stacked_rmse, axis=0)
conf_cr = 1.96*(np.std(stacked_cr, axis=0)/np.sqrt(10))
conf_prd = 1.96*(np.std(stacked_prd, axis=0)/np.sqrt(10))
conf_rmse = 1.96*(np.std(stacked_rmse, axis=0)/np.sqrt(10))
thresholds = [2, 4, 6, 8, 10]
# CIs:
plt.fill_between(x=thresholds, y1=means_cr+conf_cr, y2=means_cr-conf_cr, color="indianred", alpha=0.15)
plt.fill_between(x=thresholds, y1=means_prd+conf_prd, y2=means_prd-conf_prd, color="dodgerblue", alpha=0.15)
plt.fill_between(x=thresholds, y1=means_rmse+conf_rmse, y2=means_rmse-conf_rmse, color="orange", alpha=0.15)
# means:
plt.plot(thresholds, means_cr, "-o", color="indianred", linewidth=2, label="mean CR (with 95% CI)")
plt.plot(thresholds, means_prd, "-o", color="dodgerblue", linewidth=2, label="mean PRD (with 95% CI)")
plt.plot(thresholds, means_rmse, "-o", color="orange", linewidth=2, label="mean RMSE (with 95% CI)")
# other settings:
plt.xticks(thresholds)
plt.xlabel("thresholds [μV]")
plt.ylabel("PRD, CR, RMSE (normalized)")
plt.grid(color="silver", linestyle="--", linewidth=0.5)
plt.legend()
plt.show()

# used previously to convert tensor datasets from float64 to float32:
'''
# make float32:
for i in range(10):
    ds = torch.load(f"./data/test/tensors_{i}.pt").float()
    torch.save(ds, f"./data/float32/tensors32_{i}.pt")
'''