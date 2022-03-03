import umap
import matplotlib.pyplot as plt
import seaborn as sns

def plot_umap(name_prefix, embeddings, labels):
    sns.set(context="paper", style="white")

    reducer = umap.UMAP(random_state=42)
    embedding = reducer.fit_transform(embeddings.detach().numpy())

    fig, ax = plt.subplots(figsize=(12, 10))
    color = labels.detach().numpy()
    plt.scatter(embedding[:, 0], embedding[:, 1], c=color, cmap="Spectral", s=0.1)
    plt.setp(ax, xticks=[], yticks=[])
    plt.title("Data embedded into two dimensions by UMAP", fontsize=18)

    plt.savefig(f"visualizations/{name_prefix}_umap.png")