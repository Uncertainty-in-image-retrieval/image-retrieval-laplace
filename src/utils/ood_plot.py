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
preds_mean = np.mean(preds, axis=0)
preds_var = np.var(preds, axis=0)

reducer = umap.UMAP(random_state=42)
embedding = reducer.fit_transform(preds_mean)
print(embedding)

preds_ood = preds_ood.detach().numpy()
preds_ood_mean = np.mean(preds_ood, axis=0)

embedding_ood = reducer.transform(preds_ood_mean)
print(embedding_ood)


#means_preds_ood = preds_ood["means"].detach().numpy()
#vars_preds_ood = preds_ood["vars"].detach().numpy()
limit = 100

fig, axs = plt.subplots(2,1)

## Plot scatter with uncertainty

#axs[0].scatter(embedding[:limit,0],embedding[:limit,1], s=0.5, c="b", label="i.d.")
#axs[0].scatter(embedding_ood[:limit,0],embedding_ood[:limit,1], s=0.5, c="r", label="o.o.d")

axs[0].scatter(embedding[:,0],embedding[:,1], s=0.5, c="b", label="i.d.")
axs[0].scatter(embedding_ood[:,0],embedding_ood[:,1], s=0.5, c="r", label="o.o.d")

"""
for i in range(limit):
    elp = Ellipse((embedding[i,0],embedding[i,1]), vars_preds[i,0], vars_preds[i,1], fc='None', edgecolor='b', lw=0.5)
    axs[0].add_patch(elp)
    elp = Ellipse((embedding_ood[i,0],embedding_ood[i,1]), vars_preds_ood[i,0], vars_preds_ood[i,1], fc='None', edgecolor='r', lw=0.5)
    axs[0].add_patch(elp)

handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center')

## Plot variance density
id_density = vars_preds.flatten()
sns.kdeplot(id_density, ax=axs[1], color="b")

ood_density = vars_preds_ood.flatten()
sns.kdeplot(ood_density, ax=axs[1], color="r")
"""
plt.show()