from typing import Optional, Dict
from fab.types_ import LogProbFunc

import torch
import torch.nn as nn
import torch.nn.functional as f
from fab.target_distributions.base import TargetDistribution
from fab.utils.numerical import (
    MC_estimate_true_expectation,
    quadratic_function,
    importance_weighted_expectation,
    effective_sample_size_over_p,
    setup_quadratic_function,
)


from abc import ABC, abstractmethod
import numpy as np
import random
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
import torch.distributions as D
from torch.distributions.mixture_same_family import MixtureSameFamily

import math
from functools import partial
from scipy.optimize import linear_sum_assignment
import ot as pot

import os
from tqdm import tqdm

device = "cpu"


#######################################
def set_random_seed(SEED=0):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


set_random_seed()
#########################################################################################


class GMM(nn.Module, TargetDistribution):
    def __init__(
        self,
        dim,
        n_mixes,
        loc_scaling,
        log_var_scaling=0.1,
        seed=0,
        n_test_set_samples=1000,
        use_gpu=True,
        true_expectation_estimation_n_samples=int(1e7),
    ):
        super(GMM, self).__init__()
        self.seed = seed
        self.n_mixes = n_mixes
        self.dim = dim
        self.n_test_set_samples = n_test_set_samples

        mean = (torch.rand((n_mixes, dim)) - 0.5) * 2 * loc_scaling
        log_var = torch.ones((n_mixes, dim)) * log_var_scaling

        self.register_buffer("cat_probs", torch.ones(n_mixes))
        self.register_buffer("locs", mean)
        self.register_buffer("scale_trils", torch.diag_embed(f.softplus(log_var)))
        self.expectation_function = quadratic_function
        self.register_buffer(
            "true_expectation",
            MC_estimate_true_expectation(
                self, self.expectation_function, true_expectation_estimation_n_samples
            ),
        )
        # self.device = "cuda" if use_gpu else "cpu"
        self.device = "cpu"
        self.to(self.device)

    def to(self, device):
        if device == "cuda":
            if torch.cuda.is_available():
                self.cuda()
        else:
            self.cpu()

    @property
    def distribution(self):
        mix = torch.distributions.Categorical(self.cat_probs)
        com = torch.distributions.MultivariateNormal(
            self.locs, scale_tril=self.scale_trils, validate_args=False
        )
        return torch.distributions.MixtureSameFamily(
            mixture_distribution=mix, component_distribution=com, validate_args=False
        )

    @property
    def test_set(self) -> torch.Tensor:
        return self.sample((self.n_test_set_samples,))

    def log_prob(self, x: torch.Tensor):
        log_prob = self.distribution.log_prob(x)
        # Very low probability samples can cause issues (we turn off validate_args of the
        # distribution object which typically raises an expection related to this.
        # We manually decrease the distributions log prob to prevent them having an effect on
        # the loss/buffer.
        # mask = torch.zeros_like(log_prob)
        # mask[log_prob < -1e4] = - torch.tensor(float("inf"))
        # log_prob = log_prob + mask
        log_prob = torch.clamp(log_prob, min=-1e4)
        return log_prob

    def sample(self, shape=(1,)):
        return self.distribution.sample(shape)

    def evaluate_expectation(self, samples, log_w):
        expectation = importance_weighted_expectation(
            self.expectation_function, samples, log_w
        )
        true_expectation = self.true_expectation.to(expectation.device)
        bias_normed = (expectation - true_expectation) / true_expectation
        return bias_normed

    def performance_metrics(
        self,
        samples: torch.Tensor,
        log_w: torch.Tensor,
        log_q_fn: Optional[LogProbFunc] = None,
        batch_size: Optional[int] = None,
    ) -> Dict:
        bias_normed = self.evaluate_expectation(samples, log_w)
        bias_no_correction = self.evaluate_expectation(samples, torch.ones_like(log_w))
        if log_q_fn:
            log_q_test = log_q_fn(self.test_set)
            log_p_test = self.log_prob(self.test_set)
            test_mean_log_prob = torch.mean(log_q_test)
            kl_forward = torch.mean(log_p_test - log_q_test)
            ess_over_p = effective_sample_size_over_p(log_p_test - log_q_test)
            summary_dict = {
                "test_set_mean_log_prob": test_mean_log_prob.cpu().item(),
                "bias_normed": torch.abs(bias_normed).cpu().item(),
                "bias_no_correction": torch.abs(bias_no_correction).cpu().item(),
                "ess_over_p": ess_over_p.detach().cpu().item(),
                "kl_forward": kl_forward.detach().cpu().item(),
            }
        else:
            summary_dict = {
                "bias_normed": bias_normed.cpu().item(),
                "bias_no_correction": torch.abs(bias_no_correction).cpu().item(),
            }
        return summary_dict


def save_gmm_as_numpy(target: GMM):
    """Save params of GMM problem."""
    import pickle

    x_shift, A, b = setup_quadratic_function(torch.ones(target.dim), seed=0)
    params = {
        "mean": target.locs.numpy(),
        "scale_tril": target.scale_trils.numpy(),
        "true_expectation": target.true_expectation.numpy(),
        "expectation_x_shift": x_shift.numpy(),
        "expectation_A": A.numpy(),
        "expectation_b": b.numpy(),
    }
    with open("gmm_problem.pkl", "wb") as f:
        pickle.dump(params, f)


#################################################################
import matplotlib.pyplot as plt
from fab.utils.plotting import plot_contours, plot_marginal_pair

# torch.manual_seed(0)
loc_scaling = 40
target = GMM(dim=2, n_mixes=40, loc_scaling=40.0)


def energy_function(y):

    return -target.log_prob(y)


###############################################################
def generate_custom_gaussian_samples(x0, beta, t, batch_size_y):
    """
    Generate samples from a time-dependent multivariate Gaussian distribution.

    Parameters:
    x0 (torch.Tensor): Initial mean vectors of shape (batch_size_x, d)
    beta (float or torch.Tensor): Parameter controlling the distribution
    t (float or torch.Tensor): Time parameter
    batch_size_y (int): Number of samples to generate for each mean

    Returns:
    Tuple of (covariance, mean, samples)
    """
    # Ensure inputs are tensors
    beta = torch.tensor(beta, dtype=x0.dtype)
    t = torch.tensor(t, dtype=x0.dtype)

    # Ensure x0 is a 2D tensor
    if x0.dim() == 1:
        x0 = x0.unsqueeze(0)

    batch_size_x, d = x0.shape

    # Compute mean vector for each batch
    sqrt_beta = torch.sqrt(beta)
    cosh_term = torch.cosh((1 - t) * sqrt_beta)
    sinh_term = torch.sinh((1 - t) * sqrt_beta)
    coth_term = 1 / torch.tanh(sqrt_beta)

    mean = x0 / (cosh_term - sinh_term * coth_term)

    cov_inv = torch.zeros((d, d), dtype=x0.dtype, device=x0.device)

    for i in range(d):
        for j in range(d):
            if i == j:
                # Diagonal terms that evolve with time
                cov_inv[i, j] = sqrt_beta * (
                    1 / torch.tanh((1 - t) * sqrt_beta) - 1 / torch.tanh(sqrt_beta)
                )

    cov = torch.inverse(cov_inv)

    # Generate samples
    samples_list = []
    for mean_batch in mean:
        # Sample batch_size_y samples from this mean
        batch_samples = torch.distributions.MultivariateNormal(mean_batch, cov).sample(
            (batch_size_y,)
        )
        samples_list.append(batch_samples)

    # Concatenate samples along the first dimension
    samples = torch.stack(samples_list, dim=0)  # Modified by Minkyu
    # samples = torch.cat(samples_list, dim=0)

    return cov_inv, mean, samples


def compute_control(x, y, y_star, H_inv, beta, t, energy_function):
    """
    Computes the control vector based on the given energy function and parameters.
    Parameters:
    - x: Tensor of shape (batch_size_x, dim) representing input x.
    - y: Tensor of shape (batch_size_y, dim), sampled y values.
    - beta: Scalar representing a parameter for the equation.
    - t: Scalar time factor for the equation.
    - energy_function: A function E(y) that returns the energy for each y, shape (batch_size_y).
    Returns:
    - u: Tensor of shape (batch_size_x, dim), the computed control vector for each x.
    """
    device = x.device
    beta_sqrt = torch.sqrt(beta).to(device)
    cosh_term = torch.cosh((1 - t) * beta_sqrt).to(device)
    sinh_term = torch.sinh((1 - t) * beta_sqrt).to(device)
    coth_beta = 1 / torch.tanh(beta_sqrt).to(device)

    # Step 0: Mahalanobis distance (y - y_star)^T H_inv (y - y_star)
    diff = y - y_star.unsqueeze(1)  # Shape: (batch_size_x, batch_size_y, dim)
    mahalanobis = 0.5 * torch.einsum(
        "bnd,dd,bnd->bn", diff, H_inv.to(device), diff
    )  # Shape: (batch_size_x, batch_size_y)

    # Step 1: Compute the energy term: -E(y)
    E_y = energy_function(y)  # Shape: (batch_size_y)

    # Step 2: Compute squared terms
    x_squared = torch.sum(x**2, dim=-1, keepdim=True)  # Shape: (batch_size_x, 1)
    y_squared = torch.sum(y**2, dim=-1)  # Shape: (batch_size_y)

    # Step 3: Compute inner product
    # x_expanded = x.unsqueeze(1)  # Shape: (batch_size_x, 1, dim)
    # y_expanded = y.unsqueeze(0)  # Shape: (1, batch_size_y, dim)
    # inner_product = torch.sum(x_expanded * y_expanded, dim=-1)  # Shape: (batch_size_x, batch_size_y)
    inner_product = torch.einsum("id,ijd->ij", x, y)  # Modified by Minkyu

    # Step 4: Compute log(G-(t;x;y) / G+(1;y;0))
    # log_G_ratio = -beta_sqrt * (cosh_term * (x_squared + y_squared.unsqueeze(0)) - 2 * inner_product) / (2 * sinh_term) + \
    #               (y_squared.unsqueeze(0) * beta_sqrt / 2) * coth_beta
    log_G_ratio = (
        -beta_sqrt
        * (cosh_term * (x_squared + y_squared) - 2 * inner_product)
        / (2 * sinh_term)
        + (y_squared * beta_sqrt / 2) * coth_beta
    )  # Modified by Minkyu

    # Step 5: Combine terms for the exponent
    # exponent = -E_y.unsqueeze(0) + log_G_ratio - mahalanobis # Shape: (batch_size_x, batch_size_y)
    exponent = -E_y + log_G_ratio + mahalanobis  # Modified by Minkyu

    # Step 6: Apply softmax over the batch_size_y dimension
    w = torch.softmax(exponent, dim=1)  # Shape: (batch_size_x, batch_size_y)

    # Step 7: Compute the weighted sum of y
    # weighted_sum = torch.sum(w.unsqueeze(-1) * y.unsqueeze(0), dim=1)  # Shape: (batch_size_x, dim)
    weighted_sum = torch.sum(w.unsqueeze(-1) * y, dim=1)  # Modified by Minkyu

    # Step 8: Compute the control vector u
    u = (weighted_sum - x * cosh_term) * (
        beta_sqrt / sinh_term
    )  # Shape: (batch_size_x, dim)

    return u


# Compute 2-Wasserstein distance


def wasserstein(
    x0: torch.Tensor,
    x1: torch.Tensor,
    method: Optional[str] = None,
    reg: float = 0.05,
    power: int = 2,
    **kwargs,
) -> float:
    assert power == 1 or power == 2
    # ot_fn should take (a, b, M) as arguments where a, b are marginals and
    # M is a cost matrix
    if method == "exact" or method is None:
        ot_fn = pot.emd2
    elif method == "sinkhorn":
        ot_fn = partial(pot.sinkhorn2, reg=reg)
    else:
        raise ValueError(f"Unknown method: {method}")

    a, b = pot.unif(x0.shape[0]), pot.unif(x1.shape[0])
    if x0.dim() > 2:
        x0 = x0.reshape(x0.shape[0], -1)
    if x1.dim() > 2:
        x1 = x1.reshape(x1.shape[0], -1)
    M = torch.cdist(x0, x1)
    if power == 2:
        M = M**2
    ret = ot_fn(a, b, M.detach().cpu().numpy(), numItermax=1e7)
    if power == 2:
        ret = math.sqrt(ret)
    return ret


#######################################################################
# main loop for H-PID

eps = 1e-3
num_steps = 501
tt = torch.linspace(eps, 1 - eps, num_steps).to(device)
delta_t = tt[1] - tt[0]
beta = torch.tensor(1.0, device=device)  # Example scalar for beta
batch_size, num_samples, dim = 2000, 500, 2



x1 = target.sample((batch_size,))
y0 = x1[0:999,:]
y1 = x1[1000:-1,:]
plt.figure()
plotting_bounds = (-loc_scaling * 1.4, loc_scaling * 1.4)
plot_contours(
    target.log_prob,
    bounds=plotting_bounds,
    n_contour_levels=50,
    grid_width_n_points=200,
)

plt.plot(y0[:, 0].cpu().numpy(), y0[:, 1].cpu().numpy(), "o", color="red", alpha=0.5)
#plt.plot(y1[:, 0].cpu().numpy(), y1[:, 1].cpu().numpy(), "o", color="blue", alpha=0.5)

os.makedirs("figs", exist_ok=True)
plt.savefig(f"figs/target.png")
plt.close()

#print("generated sample : ", x0)
#print("target sample : ", x1)

w2 = wasserstein(y0, y1, power=2)
print("2-Wasserstein of H-PID : ", w2)
