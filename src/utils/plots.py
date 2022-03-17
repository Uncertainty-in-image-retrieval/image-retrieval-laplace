import umap.umap_ as umap
import matplotlib.pyplot as plt
import seaborn as sns

def plot_umap(name_prefix, embeddings, labels):
    sns.set(context="paper", style="white")

    reducer = umap.UMAP(random_state=42)
    embedding = reducer.fit_transform(embeddings.detach().numpy())

    fig, ax = plt.subplots(figsize=(12, 10))
    color = labels.detach().numpy()
    plt.scatter(embedding[:, 0], embedding[:, 1], label=color, c=color, s=1)
    plt.setp(ax, xticks=[], yticks=[])
    plt.legend()
    plt.title("Data embedded into two dimensions by UMAP", fontsize=18)

    plt.savefig(f"visualizations/{name_prefix}_umap.png")