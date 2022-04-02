import torch

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