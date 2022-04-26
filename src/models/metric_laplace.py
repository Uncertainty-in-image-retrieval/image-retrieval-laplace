import torch
from laplace.baselaplace import DiagLaplace
from pytorch_metric_learning import losses
from pytorch_metric_learning import miners
from torch.nn.utils import parameters_to_vector
from asdfghjkl.gradient import batch_gradient
from laplace.curvature.asdl import _get_batch_grad
from tqdm import tqdm
from laplace.utils import get_nll


class ContrastiveHessianCalculator:
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

    def calculate_hessian(self, *inputs, model, num_outputs, hessian_structure="diag", agg="sum"):
        x1 = inputs[0]
        x2 = inputs[1]
        print("Computinng the jacobians...")
        Jz1, f1 = self.jacobians(x1, model, output_size=num_outputs)
        Jz2, f2 = self.jacobians(x2, model, output_size=num_outputs)

        y = inputs[2]

        print("Actual Hessian computation...")
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
        elif agg == 'mean':
            Hs = Hs.mean(dim=0)

        return Hs


class MetricLaplace(DiagLaplace):
    def __init__(self, model, likelihood):
        super().__init__(model, likelihood)
        
        self.loss_fn = losses.ContrastiveLoss()
        self.hessian_calculator = ContrastiveHessianCalculator()
        self.miner = miners.BatchEasyHardMiner(pos_strategy='easy', neg_strategy='easy',
                                               allowed_pos_range=(0.2, 1),
                                               allowed_neg_range=(0.2, 1))

    def _mine(self, X, y):
        a1, p, a2, n = self.miner(X, y)
   
        return a1, p, a2, n
    
    def _curv_closure(self, X, y, N):
        embeddings = self.model(X)
        
        a1, p, a2, n = self._mine(embeddings, y)
        
        loss = self.loss_fn(embeddings, y, (a1, p, a2, n))
                
        x1 = X[torch.cat((a1, a2))]
        x2 = X[torch.cat((p, n))]
        t = torch.cat((torch.ones(p.shape[0]), torch.zeros(n.shape[0])))

        H = self.hessian_calculator.calculate_hessian(x1, x2, t, model=self.model, num_outputs=embeddings.shape[-1])
        
        return loss, H
    
    def fit(self, train_loader, override=True):
        """Fit the local Laplace approximation at the parameters of the model.
        Parameters
        ----------
        train_loader : torch.data.utils.DataLoader
            each iterate is a training batch (X, y);
            `train_loader.dataset` needs to be set to access \\(N\\), size of the data set
        override : bool, default=True
            whether to initialize H, loss, and n_data again; setting to False is useful for
            online learning settings to accumulate a sequential posterior approximation.
        """
        if override:
            self._init_H()
            self.loss = 0
            self.n_data = 0

        self.model.eval()
        self.mean = parameters_to_vector(self.model.parameters()).detach()

        X, _ = next(iter(train_loader))
        with torch.no_grad():
            try:
                out = self.model(X[:1].to(self._device))
            except (TypeError, AttributeError):
                out = self.model(X.to(self._device))
        self.n_outputs = out.shape[-1]
        setattr(self.model, 'output_size', self.n_outputs)

        N = len(train_loader.dataset)
        for X, y in tqdm(train_loader):
            self.model.zero_grad()
            X, y = X.to(self._device), y.to(self._device)
            loss_batch, H_batch = self._curv_closure(X, y, N)
            self.loss += loss_batch
            self.H += H_batch

        self.n_data += N

    def optimize_prior_precision_base(self, pred_type, method='marglik', n_steps=100, lr=1e-1,
                                      init_prior_prec=1., val_loader=None, loss=get_nll,
                                      log_prior_prec_min=-4, log_prior_prec_max=4, grid_size=100,
                                      link_approx='probit', n_samples=100, verbose=False,
                                      cv_loss_with_var=False):
        """Optimize the prior precision post-hoc using the `method`
        specified by the user.
        Parameters
        ----------
        pred_type : {'glm', 'nn', 'gp'}, default='glm'
            type of posterior predictive, linearized GLM predictive or neural
            network sampling predictive or Gaussian Process (GP) inference.
            The GLM predictive is consistent with the curvature approximations used here.
        method : {'marglik', 'CV'}, default='marglik'
            specifies how the prior precision should be optimized.
        n_steps : int, default=100
            the number of gradient descent steps to take.
        lr : float, default=1e-1
            the learning rate to use for gradient descent.
        init_prior_prec : float, default=1.0
            initial prior precision before the first optimization step.
        val_loader : torch.data.utils.DataLoader, default=None
            DataLoader for the validation set; each iterate is a training batch (X, y).
        loss : callable, default=get_nll
            loss function to use for CV.
        cv_loss_with_var: bool, default=False
            if true, `loss` takes three arguments `loss(output_mean, output_var, target)`,
            otherwise, `loss` takes two arguments `loss(output_mean, target)`
        log_prior_prec_min : float, default=-4
            lower bound of gridsearch interval for CV.
        log_prior_prec_max : float, default=4
            upper bound of gridsearch interval for CV.
        grid_size : int, default=100
            number of values to consider inside the gridsearch interval for CV.
        link_approx : {'mc', 'probit', 'bridge'}, default='probit'
            how to approximate the classification link function for the `'glm'`.
            For `pred_type='nn'`, only `'mc'` is possible.
        n_samples : int, default=100
            number of samples for `link_approx='mc'`.
        verbose : bool, default=False
            if true, the optimized prior precision will be printed
            (can be a large tensor if the prior has a diagonal covariance).
        """
        if method == 'marglik':
            self.prior_precision = init_prior_prec
            log_prior_prec = self.prior_precision.log()
            log_prior_prec.requires_grad = True
            optimizer = torch.optim.Adam([log_prior_prec], lr=lr)
            for _ in tqdm(range(n_steps)):
                optimizer.zero_grad()
                prior_prec = log_prior_prec.exp()
                neg_log_marglik = -self.log_marginal_likelihood(prior_precision=prior_prec)
                neg_log_marglik.backward(retain_graph=True)
                optimizer.step()
            self.prior_precision = log_prior_prec.detach().exp()
        elif method == 'CV':
            if val_loader is None:
                raise ValueError('CV requires a validation set DataLoader')
            interval = torch.logspace(
                log_prior_prec_min, log_prior_prec_max, grid_size
            )
            self.prior_precision = self._gridsearch(
                loss, interval, val_loader, pred_type=pred_type,
                link_approx=link_approx, n_samples=n_samples, loss_with_var=cv_loss_with_var
            )
        else:
            raise ValueError('For now only marglik and CV is implemented.')
        if verbose:
            print(f'Optimized prior precision is {self.prior_precision}.')