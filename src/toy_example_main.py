import time
import torch
from laplace import Laplace
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.main import preproc_data

#from src.models.manual_hessian import RmseHessianCalculator

def compute_hessian(x, feature_maps, net, output_size, h_scale):
    H = []
    bs = x.shape[0]
    feature_maps = [x] + feature_maps
    tmp = torch.diag_embed(torch.ones(bs, output_size, device=x.device))

    with torch.no_grad():
        for k in range(len(net) - 1, -1, -1):

            # compute Jacobian wrt input
            if isinstance(net[k], torch.nn.Linear):
                diag_elements = torch.diagonal(tmp, dim1=1, dim2=2)
                feature_map_k2 = (feature_maps[k] ** 2).unsqueeze(1)

                h_k = torch.bmm(diag_elements.unsqueeze(2), feature_map_k2).view(bs, -1)

                # has a bias
                if net[k].bias is not None:
                    h_k = torch.cat([h_k, diag_elements], dim=1)

                H = [h_k] + H

            # compute Jacobian wrt input
            if isinstance(net[k], torch.nn.Tanh):
                J_tanh = torch.diag_embed(
                    torch.ones(feature_maps[k + 1].shape, device=x.device)
                    - feature_maps[k + 1] ** 2
                )
                tmp = torch.einsum("bnm,bnj,bjk->bmk", J_tanh, tmp, J_tanh)
            #elif isinstance(net[k], torch.nn.ReLU):
               # J_relu = torch.diag_embed(
              #      (feature_maps[k] > 0).float()
               # )
               # tmp = torch.einsum("bnm,bnj,bjk->bmk", J_relu, tmp, J_relu)

            if k == 0:
                break

            # compute Jacobian wrt weight
            if isinstance(net[k], torch.nn.Linear):
                tmp = torch.einsum("nm,bnj,jk->bmk",
                                   net[k].weight,
                                   tmp,
                                   net[k].weight)

    H = torch.cat(H, dim=1)

    # mean over batch size scaled by the size of the dataset
    H = h_scale * torch.mean(H, dim=0)

    return H


num_observations = 1000

X = torch.rand((num_observations, 1)).float()
y = 4.5 * torch.cos(2 * torch.pi * X + 1.5 * torch.pi) - \
    3 * torch.sin(4.3 * torch.pi * X + 0.3 * torch.pi) + \
    3.0 * X - 7.5

#train_loader, test_loader, train_data, test_data = preproc_data()
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=num_observations)
x, y = next(iter(dataloader))


model = nn.Sequential(
    nn.Linear(1, 16),
    nn.ReLU(),
    nn.Linear(16, 16),
    nn.ReLU(),
    nn.Linear(16, 32),
    nn.ReLU(),
    nn.Linear(32, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 16),
    nn.ReLU(),
    nn.Linear(16, 1)
)
num_params = sum(p.numel() for p in model.parameters())

for hessian_structure in ["diag"]:#, "full"]:
    la = Laplace(
        model,
        "regression",
        hessian_structure=hessian_structure,
        subset_of_weights="all",
    )
    t0 = time.perf_counter()
    la.fit(dataloader)
    elapsed_la = time.perf_counter() - t0

    activation = []
    def get_activation():
        def hook(model, input, output):
            activation.append(output.detach())
        return hook
    for layer in model:
        layer.register_forward_hook(get_activation())
    output = model(x)

    t0 = time.perf_counter()
    Hs = compute_hessian(x, activation, model, output_size=1,
                         h_scale=num_observations)
    elapsed = time.perf_counter() - t0

    torch.testing.assert_close(la.H, Hs, rtol=1e-2, atol=0.)  # Less than 1% off

    # Prior precision is one.
    if hessian_structure == "diag":
        prior_precision = torch.ones((num_params,))  # Prior precision is one.
        precision = Hs + prior_precision
        covariance_matrix = 1 / precision
        torch.testing.assert_close(la.posterior_variance, covariance_matrix, rtol=1e-5, atol=0.)
    elif hessian_structure == "full":
        prior_precision = torch.eye(num_params)
        precision = Hs + prior_precision
        covariance_matrix = torch.inverse(precision)
        torch.testing.assert_close(la.posterior_covariance, covariance_matrix, rtol=1e-1, atol=0.)
    else:
        raise NotImplementedError

