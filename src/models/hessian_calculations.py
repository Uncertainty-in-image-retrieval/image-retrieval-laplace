import torch
from asdfghjkl.gradient import batch_gradient
from laplace.curvature.asdl import _get_batch_grad


class HessianCalculator:
    def calculate_hessian(self, *inputs, model, num_outputs, hessian_structure="diag", agg="sum"):
        raise NotImplementedError

    def jacobians(self, x, model, output_size=784):
        """Compute Jacobians \\(\\nabla_\\theta f(x;\\theta)\\) at current parameter \\(\\theta\\)
        using asdfghjkl's gradient per output dimension.
        Parameters
        ----------
        x : torch.Tensor
            input data `(batch, input_shape)` on compatible device with model.
        Returns
        -------
        Js : torch.Tensor
            Jacobians `(batch, parameters, outputs)`
        f : torch.Tensor
            output function `(batch, outputs)`
        """
        Js = list()
        for i in range(output_size):
            def loss_fn(outputs, targets):
                return outputs[:, i].sum()

            f = batch_gradient(model, loss_fn, x, None).detach()
            Jk = _get_batch_grad(model)

            Js.append(Jk)
        Js = torch.stack(Js, dim=1)

        return Js, f


class RmseHessianCalculator(HessianCalculator):
    def calculate_hessian(self, *inputs, model, num_outputs, hessian_structure="diag", agg="sum"):
        x = inputs[0]
        Js, f = self.jacobians(x, model, output_size=num_outputs)

        if hessian_structure == "diag":
            Hs = torch.einsum("nij,nij->nj", Js, Js)
        elif hessian_structure == "full":
            Hs = torch.einsum("nij,nkl->njl", Js, Js)
        else:
            raise NotImplementedError

        if agg == "sum":
            Hs = Hs.sum(dim=0)

        return Hs


class ContrastiveHessianCalculator(HessianCalculator):
    def calculate_hessian(self, *inputs, model, num_outputs, hessian_structure="diag", agg="sum"):
        x1 = inputs[0]
        x2 = inputs[1]
        Jz1, f1 = self.jacobians(x1, model, output_size=num_outputs)
        Jz2, f2 = self.jacobians(x2, model, output_size=num_outputs)

        y = inputs[2]

        # L = y * ||z_1 - z_2||^2 + (1 - y) max(0, m - ||z_1 - z_2||^2)
        # The Hessian is equal to Hs, except when we have:
        # 1. A negative pair
        # 2. The margin minus the norm is negative
        m = 0.1  # margin
        mask = torch.logical_and(
            (1 - y).bool(),
            m - torch.einsum("no,no->n", f1 - f2, f1 - f2) < 0
            # m - torch.pow(torch.linalg.norm(f1 - f2, dim=1), 2) < 0
        )
        if hessian_structure == "diag":
            Hs = torch.einsum("nij,nij->nj", Jz1, Jz1) + \
                 torch.einsum("nij,nij->nj", Jz2, Jz2) - \
                 2 * (
                         torch.einsum("nij,nij->nj", Jz1, Jz2) +
                         torch.einsum("nij,nij->nj", Jz2, Jz1)
                 )
            mask = mask.view(-1, 1).expand(*Hs.shape)
        elif hessian_structure == "full":
            Hs = torch.einsum("nij,nkl->njl", Jz1, Jz1) + \
                 torch.einsum("nij,nkl->njl", Jz2, Jz2) - \
                 2 * (
                         torch.einsum("nij,nkl->njl", Jz1, Jz2) +
                         torch.einsum("nij,nkl->njl", Jz2, Jz1)
                 )
            mask = mask.view(-1, 1, 1).expand(*Hs.shape)
        else:
            raise NotImplementedError

        Hs = Hs.masked_fill_(mask, 0.)

        if agg == "sum":
            Hs = Hs.sum(dim=0)

        return Hs