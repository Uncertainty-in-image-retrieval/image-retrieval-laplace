import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
# matplotlib.use('Qt5Agg')
import pickle
import numpy as np
import seaborn as sns
import umap.umap_ as umap

with open("pred_in.pkl", "rb") as f:
    preds = pickle.load(f)
with open("pred_ood.pkl", "rb") as f:
    preds_ood = pickle.load(f)

preds = preds.detach().numpy()
preds_flatten = np.reshape(preds, (100*10000, 16))[:16*10000,:]

preds_ood = preds_ood.detach().numpy()
preds_ood_flatten = np.reshape(preds_ood, (100*10000, 16))[:16*10000,:]


"""
reducer = umap.UMAP(random_state=42, metric="precomputed")
preds_reduced = preds_flatten[:20000,:]
from sklearn.metrics.pairwise import euclidean_distances
print("computing shit")
dist_matrix = euclidean_distances(preds_reduced, preds_reduced)
print(dist_matrix)
print(dist_matrix.shape)
import time 
start = time.time()
embeddings_flatten = reducer.fit(dist_matrix)
print("Time: ", str(time.time()-start))
embeddings = np.reshape(embeddings_flatten, (8, 10000, 2))
exit()
"""

reducer = umap.UMAP(random_state=42)
print("Fitting")
import time 
start = time.time()
reducer.fit(preds_flatten[:10000,:])
print("Time: ", str(time.time()-start))
embeddings_flatten = reducer.transform(preds_flatten)
embeddings = np.reshape(embeddings_flatten, (16, 10000, 2))

preds_in_mean = np.mean(embeddings, axis=0)
preds_in_var = np.var(embeddings, axis=0)

embeddings_ood_flatten = reducer.transform(preds_ood_flatten)
embeddings_ood = np.reshape(embeddings_ood_flatten, (16, 10000, 2))

preds_ood_mean = np.mean(embeddings_ood, axis=0)
preds_ood_var = np.var(embeddings_ood, axis=0)


limit = 100

fig, axs = plt.subplots(2,1)

## Plot scatter with uncertainty

axs[0].scatter(preds_in_mean[:limit,0],preds_in_mean[:limit,1], s=0.5, c="b", label="i.d.")
axs[0].scatter(preds_ood_mean[:limit,0],preds_ood_mean[:limit,1], s=0.5, c="r", label="o.o.d")

#axs[0].scatter(embeddings[:,0],embeddings[:,1], s=0.5, c="b", label="i.d.")
#axs[0].scatter(embedding_ood[:,0],embedding_ood[:,1], s=0.5, c="r", label="o.o.d")


for i in range(limit):
    elp = Ellipse((preds_in_mean[i,0],preds_in_mean[i,1]), preds_in_var[i,0], preds_in_var[i,1], fc='None', edgecolor='b', lw=0.5)
    axs[0].add_patch(elp)
    elp = Ellipse((preds_ood_mean[i,0],preds_ood_mean[i,1]), preds_ood_var[i,0], preds_ood_var[i,1], fc='None', edgecolor='r', lw=0.5)
    axs[0].add_patch(elp)

handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center')

## Plot variance density
id_density = preds_in_var.flatten()
sns.kdeplot(id_density, ax=axs[1], color="b")

ood_density = preds_ood_var.flatten()
sns.kdeplot(ood_density, ax=axs[1], color="r")

plt.savefig("visualizations/in_out_sample_distrib_16networks_limit.png")

plt.show()