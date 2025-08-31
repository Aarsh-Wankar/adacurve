import torch
from torch.optim import Optimizer

class AdaCurve(Optimizer):
    """
    NOTE:
      - You MUST call loss.backward(create_graph=True) to enable Hessian-vector products.
      - Hutchinson’s method is used to estimate the diagonal of the Hessian.
      - AdamW-style decoupled weight decay is supported.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 curvature_iters=1, hessian_update_every=1, weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        curvature_iters=curvature_iters,
                        hessian_update_every=hessian_update_every,
                        weight_decay=weight_decay)
        super(AdaCurve, self).__init__(params, defaults)

        self._step_count = 0
        self._cached_curvatures = None

    def _get_all_params(self):
        """Flatten all parameters with gradients."""
        return [p for group in self.param_groups for p in group['params'] if p.grad is not None]

    def _hess_vec_prod(self, params, vec):
        """Compute Hessian-vector product H*v using autograd."""
        grads = [p.grad for p in params]
        grad_dot_vec = sum(torch.dot(g.view(-1), v.view(-1)) for g, v in zip(grads, vec))
        hvp = torch.autograd.grad(grad_dot_vec, params, retain_graph=True)
        return hvp

    def _compute_curvature(self, params, iters):
        """Estimate Hessian diagonal with Hutchinson’s method."""
        device = params[0].device
        curvature_array = [torch.zeros_like(p, device=device) for p in params]

        for _ in range(iters):
            # Random Rademacher vector (+1/-1)
            rademacher_vec = [(torch.randint_like(p, high=2, device=device) * 2 - 1).float()
                              for p in params]

            hvp = self._hess_vec_prod(params, rademacher_vec)
            for c, h, v in zip(curvature_array, hvp, rademacher_vec):
                if h is not None:
                    c.addcmul_(h, v)  # elementwise (Hv) ⊙ v

        # Average over Hutchinson samples
        return [c.div_(iters) for c in curvature_array]

    @torch.no_grad()
    def step(self, closure=None):
        """Perform one optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        params_with_grad = self._get_all_params()
        if not params_with_grad:
            return loss

        self._step_count += 1
        group = self.param_groups[0]  # assume single group for simplicity
        weight_decay = group.get('weight_decay', 0.0)

        # Compute Hessian diag every `hessian_update_every` steps
        if self._step_count % group['hessian_update_every'] == 0 or self._cached_curvatures is None:
            with torch.enable_grad():
                self._cached_curvatures = self._compute_curvature(
                    params_with_grad, group['curvature_iters']
                )

        curvature_array = self._cached_curvatures

        beta1, beta2 = group['betas']
        lr = group['lr']
        eps = group['eps']

        for p, c in zip(params_with_grad, curvature_array):
            grad = p.grad
            state = self.state[p]

            if len(state) == 0:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)

            exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
            state['step'] += 1
            t = state['step']

            # Update first moment
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            # Update second moment with ABSOLUTE curvature (not squared!)
            exp_avg_sq.mul_(beta2).add_(c.abs() * grad.abs(), alpha=1 - beta2)

            # Bias corrections
            bias_correction1 = 1 - beta1 ** t
            bias_correction2 = 1 - beta2 ** t

            step_size = lr / bias_correction1
            denom = exp_avg_sq.div(bias_correction2).sqrt().add_(eps)

            # AdamW-style decoupled weight decay
            if weight_decay > 0:
                p.add_(p, alpha=-lr * weight_decay)

            # Parameter update
            p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
