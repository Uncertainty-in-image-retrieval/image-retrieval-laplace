import torch
from pl_bolts.datamodules import CIFAR10DataModule
from pytorch_metric_learning import miners
from torch import nn
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu

from src.models.hessian_calculations import ContrastiveHessianCalculator
from src.models.laplace_metric import MetricDiagLaplace
import pickle

batch_size = 8
latent_size = 3

model = nn.Sequential(
    nn.Conv2d(3, 16, 3, 1),
    nn.ReLU(),
    nn.Conv2d(16, 32, 3, 1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Dropout2d(0.25),
    nn.Flatten(),
    nn.Linear(6272, latent_size),
)
num_params = sum(p.numel() for p in model.parameters())

miner = miners.BatchEasyHardMiner(pos_strategy="all", neg_strategy="all")

data = CIFAR10DataModule("./data", batch_size=batch_size, num_workers=0, normalize=True)
data.setup()



la = MetricDiagLaplace(
    model,
    "regression",
    # subset_of_weights="last_layer",
    # hessian_structure="diag"
)

with open('models/laplace_metric.pkl', 'rb') as file:
    la = pickle.load(file)

    
predictions = la(next(iter(data.test_dataloader()))[0], pred_type='glm', link_approx='probit')
print(predictions)
print(la)

def plot_latent_space_ood(z_mu, z_sigma):
    # path, z_mu, z_sigma, labels, ood_z_mu, ood_z_sigma, ood_labels
# ):

    import numpy as np 
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse

    max_ = np.max(z_sigma)
    min_ = np.min(z_sigma)
    
    # normalize sigma
    z_sigma = ((z_sigma - min_) / (max_ - min_)) * 1

    fig, ax = plt.subplots(1, 1, figsize=(9, 9))
    for i, (z_mu_i, z_sigma_i) in enumerate(zip(z_mu, z_sigma)):

        ax.scatter(z_mu_i[0], z_mu_i[1], color="b")
        ellipse = Ellipse(
            (z_mu_i[0], z_mu_i[1]),
            width=z_sigma_i[0],
            height=z_sigma_i[1],
            fill=False,
            edgecolor="blue",
        )
        ax.add_patch(ellipse)

        if i > 500:
            ax.scatter(z_mu_i[0], z_mu_i[1], color="b", label="ID")
            break