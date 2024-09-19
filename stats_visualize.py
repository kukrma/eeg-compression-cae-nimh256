# =========================================================================== #
# author:  Martin KUKRÁL                                                      #
# date:    September 19, 2024                                                 #
# Python:  3.11.4                                                             #
# licence: CC BY-NC 4.0                                                       #
# purpose: 1) visualize results of correlation analysis                       #
#          2) apply UPGMA and visualize the results                           #
#          3) visualize results of autocorrelation analysis                   #
#          4) visualize cube root function                                    #
# =========================================================================== #
import numpy as np                       # 1.25.2
import matplotlib.pyplot as plt          # 3.7.2
import seaborn as sns                    # 0.13.0
import scipy.cluster.hierarchy as hier   # 1.11.2





# --- 1) Correlation ----------------------------------------------------------
# load correlation data:
corrs_load = np.load("./outputs/correlations.npy") #load correlation matrices from file
means = np.mean(corrs_load, axis=0) #mean across subjects
conf = 1.96*(np.std(corrs_load, axis=0)/np.sqrt(25)) #95% CI across subjects
# heatmap of correlation matrix for 1st subject:
heat = sns.heatmap(corrs_load[0], cmap="RdYlBu_r", annot=False, center=0, square=True, mask=np.triu(means))
heat.set_xticks([])
heat.set_yticks([])
plt.xlabel("channels")
plt.ylabel("channels")
plt.show()
# heatmap of correlation mean across subjects:
heat = sns.heatmap(means, cmap="RdYlBu_r", annot=False, center=0, square=True, mask=np.triu(means))
heat.set_xticks([])
heat.set_yticks([])
plt.xlabel("channels")
plt.ylabel("channels")
plt.show()
# heatmap of 95% correlation CI across subjects:
heat = sns.heatmap(conf, cmap="BuGn", annot=False, square=True, mask=np.triu(conf), vmin=0)
heat.set_xticks([])
heat.set_yticks([])
plt.xlabel("channels")
plt.ylabel("channels")
plt.show()

# --- 2) UPGMA clustering -----------------------------------------------------
# calculate the UPGMA:
linked = hier.linkage(means, "average")
dendro = hier.dendrogram(linked, no_plot=True)
order = dendro["leaves"]
# save the new order of channels as file:
np.save("./outputs/reorder.npy", order)
# visualize the UPGMA results:
reordered = means[order, :][:, order]
clust = sns.clustermap(reordered, cmap="RdYlBu_r", annot=False, center=0, mask=np.triu(means), col_cluster=False)
clust.ax_heatmap.set_xticks([])
clust.ax_heatmap.set_yticks([])
clust.ax_heatmap.set_xlabel("channels")
clust.ax_heatmap.set_ylabel("channels")
clust.ax_heatmap.set_aspect("equal", adjustable="box")
plt.show()

# --- 3) Autocorrelation ------------------------------------------------------
# load ACF data:
acf_load = np.load("./outputs/autocorrelations.npy")
means = np.mean(acf_load, axis=0)[order, :] #mean across subjects
conf = 1.96*(np.std(acf_load, axis=0)/np.sqrt(25)) #95% CI across subjects
# pixel map of ACF mean:
heat = sns.heatmap(means, cmap="RdYlBu_r", annot=False, center=0)
heat.set_xticks([500, 1000, 1500, 2000, 2500, 3000])
heat.set_xticklabels(["500", "1000", "1500", "2000", "2500", "3000"])
heat.set_xticklabels(heat.get_xticks(), size = 8)
heat.set_yticks([])
plt.xlabel("lags")
plt.ylabel("channels")
plt.show()
# pixel map of ACF 95% CI across subjects:
heat = sns.heatmap(conf, cmap="BuGn", annot=False, vmin=0)
heat.set_xticks([500, 1000, 1500, 2000, 2500, 3000])
heat.set_xticklabels(["500", "1000", "1500", "2000", "2500", "3000"])
heat.set_xticklabels(heat.get_xticks(), size = 8)
heat.set_yticks([])
plt.xlabel("lags")
plt.ylabel("channels")
plt.show()

'''
### REMOVED FROM THE FINAL VERSION OF THE ARTICLE ###
# --- 4) Cube root plot -------------------------------------------------------
vals = np.linspace(-4, 4, 1000)
cube = np.cbrt(vals)
plt.axhline(0, color="black", linewidth=0.5)
plt.axvline(0, color="black", linewidth=0.5)
plt.plot(vals, vals, color="green", label="linear function")
plt.plot(vals, cube, color="red", label="cube root")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.xlim(-4, 4)
plt.ylim(-4, 4)
plt.legend()
plt.axis("equal")
plt.grid(color = 'grey', linestyle = '--', linewidth = 0.5)
plt.show()
'''