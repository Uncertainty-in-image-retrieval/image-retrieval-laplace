from src.models.model import VGG, Net, NetSoftmax
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

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

            if k == 0:
                break

            # compute Jacobian wrt weight
            if isinstance(net[k], torch.nn.Linear):
                tmp = torch.einsum("nm,bnj,jk->bmk", net[k].weight, tmp, net[k].weight)

    H = torch.cat(H, dim=1)

    # mean over batch size scaled by the size of the dataset
    H = h_scale * torch.mean(H, dim=0)

    return H


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Init")
num_observations = 1000

X = torch.rand((num_observations, 1, 28, 28)).float()
y = 4.5 * torch.cos(2 * torch.pi * X + 1.5 * torch.pi) - \
    3 * torch.sin(4.3 * torch.pi * X + 0.3 * torch.pi) + \
    3.0 * X - 7.5

dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=num_observations)
x, y = next(iter(dataloader))
dataloader = DataLoader(dataset, batch_size=32)

model = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
            nn.Flatten(1),
            nn.Linear(9216, 256),
            nn.ReLU(),
            nn.Dropout2d(0.25),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout2d(0.25),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Dropout2d(0.25),
            nn.Linear(32,2)
        )

#model.load_state_dict(torch.load('temp/tensor.pt'))

print("Shit 1")

activation = []
def get_activation():
    def hook(model, input, output):
        activation.append(output.detach())
    return hook
i=0
for layer in model:
    layer.register_forward_hook(get_activation())
output = model(x)
exit()
print("shit 2")
Hs = compute_hessian(x, activation, model, output_size=1,
                        h_scale=num_observations)

print("size")
print(output.size)

print(Hs.size)