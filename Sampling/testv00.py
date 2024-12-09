import abc
import numpy as np
import random 
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
import torch.distributions as D
from torch.distributions.mixture_same_family import MixtureSameFamily

device = 'cpu'
#######################################
def set_random_seed(SEED=1234):
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
def nll_unit_gaussian(data, sigma=1.0):
    data = data.view(data.shape[0], -1)
    loss = 0.5 * np.log(2 * np.pi) + np.log(sigma) + 0.5 * data * data / (sigma ** 2)
    return torch.sum(torch.flatten(loss, start_dim=1), -1)


class BaseSet(abc.ABC, Dataset):
    def __init__(self, len_data=-2333):
        self.num_sample = len_data
        self.data = None
        self.data_ndim = None
        self._gt_ksd = None

    def gt_logz(self):
        raise NotImplementedError

    @abc.abstractmethod
    def energy(self, x):
        return

    def unnorm_pdf(self, x):
        return torch.exp(-self.energy(x))

    # hmt stands for hamiltonian
    def hmt_energy(self, x):
        dim = x.shape[-1]
        x, v = torch.split(x, dim // 2, dim=-1)
        neg_log_p_x = self.sample_energy_fn(x)
        neg_log_p_v = nll_unit_gaussian(v)
        return neg_log_p_x + neg_log_p_v

    @property
    def ndim(self):
        return self.data_ndim

    def sample(self, batch_size):
        del batch_size
        raise NotImplementedError

    def score(self, x):
        with torch.no_grad():
            copy_x = x.detach().clone()
            copy_x.requires_grad = True
            with torch.enable_grad():
                self.energy(copy_x).sum().backward()
                lgv_data = copy_x.grad.data
            return lgv_data

    def log_reward(self, x):
        return -self.energy(x)

    def hmt_score(self, x):
        with torch.no_grad():
            copy_x = x.detach().clone()
            copy_x.requires_grad = True
            with torch.enable_grad():
                self.hmt_energy(copy_x).sum().backward()
                lgv_data = copy_x.grad.data
            return lgv_data

class TwentyFiveGaussianMixture(BaseSet):
    def __init__(self, device, dim=2):
        super().__init__()
        self.data = torch.tensor([0.0])
        self.device = device

        modes = torch.Tensor([(a, b) for a in [-10, -5, 0, 5, 10] for b in [-10, -5, 0, 5, 10]]).to(self.device)

        nmode = 25
        self.nmode = nmode

        self.data_ndim = dim

        self.gmm = [D.MultivariateNormal(loc=mode.to(self.device),
                                         covariance_matrix=0.3 * torch.eye(self.data_ndim, device=self.device))
                    for mode in modes]

    def gt_logz(self):
        return 0.

    def energy(self, x):
        log_prob = torch.logsumexp(torch.stack([mvn.log_prob(x) for mvn in self.gmm]), dim=0,
                           keepdim=False) - torch.log(torch.tensor(self.nmode, device=self.device))
        return -log_prob

    def sample(self, batch_size):
        samples = torch.cat([mvn.sample((batch_size // self.nmode,)) for mvn in self.gmm], dim=0).to(self.device)
        return samples

    def viz_pdf(self, fsave="25gmm-density.png"):
        x = torch.linspace(-15, 15, 100).to(self.device)
        y = torch.linspace(-15, 15, 100).to(self.device)
        X, Y = torch.meshgrid(x, y)
        x = torch.stack([X.flatten(), Y.flatten()], dim=1)  # ?

        density = self.unnorm_pdf(x)
        return x, density

    def __getitem__(self, idx):
        del idx
        return self.data[0]
    

class NineGaussianMixture(BaseSet):
    def __init__(self, device, scale=0.5477222, dim=2):
        super().__init__()
        self.device = device
        self.data = torch.tensor([0.0])
        self.data_ndim = 2

        mean_ls = [
            [-5., -5.], [-5., 0.], [-5., 5.],
            [0., -5.], [0., 0.], [0., 5.],
            [5., -5.], [5., 0.], [5., 5.],
        ]
        nmode = len(mean_ls)
        mean = torch.stack([torch.tensor(xy) for xy in mean_ls])
        comp = D.Independent(D.Normal(mean.to(self.device), torch.ones_like(mean).to(self.device) * scale), 1)
        mix = D.Categorical(torch.ones(nmode).to(self.device))
        self.gmm = MixtureSameFamily(mix, comp)
        self.data_ndim = dim

    def gt_logz(self):
        return 0.

    def energy(self, x):
        return -self.gmm.log_prob(x).flatten()

    def sample(self, batch_size):
        return self.gmm.sample((batch_size,))

    def viz_pdf(self, fsave="ou-density.png"):
        x = torch.linspace(-8, 8, 100).to(self.device)
        y = torch.linspace(-8, 8, 100).to(self.device)
        X, Y = torch.meshgrid(x, y)
        x = torch.stack([X.flatten(), Y.flatten()], dim=1)  # ?

        density = self.unnorm_pdf(x)
        return x, density

    def __getitem__(self, idx):
        del idx
        return self.data[0]
##########################################################################
gmm25 = TwentyFiveGaussianMixture(device=device, dim=2)
gmm09 = NineGaussianMixture(device=device, dim=2)


# Generate 512 samples
#samples = gmm09.sample(512)
#plt.plot(samples[:,0].cpu().numpy(), samples[:,1].cpu().numpy(), 'o', color='blue')
#plt.savefig('original.pdf')


def normal_is(x, beta, t, num_samples):
    """
    Generates `num_samples` for each sample in the input batch `x` using
    a multivariate normal distribution with computed means and covariances.

    Parameters:
    - x: tensor of shape (batch_size, dim) where batch_size is the number of input samples.
    - beta: scalar value for covariance calculation.
    - t: scalar value for time.
    - num_samples: the number of new samples to generate for each `x` in the batch.

    Returns:
    - covar_inv: inverse covariance matrix for each sample (batch_size, dim, dim).
    - y_star: mean tensor used for generating new samples (batch_size, dim).
    - y: tensor of shape (batch_size, num_samples, dim) with generated samples.
    """
    # Ensure the code runs on the same device as x (either CPU or CUDA)
    device = x.device

    batch_size, d = x.shape  # batch_size is the number of x samples, d is dimensionality
    beta_sqrt = torch.sqrt(beta).to(device)

    # Compute coth terms
    coth1 = 1 / torch.tanh(beta_sqrt)
    coth2 = 1 / torch.tanh((1 - t) * beta_sqrt)

    # Compute coefficients and covariance
    coeff = beta_sqrt * (coth2 - coth1)
    H_inv = torch.eye(d, device=device) * coeff  # Inverse covariance matrix
    H_sqrt = torch.eye(d, device=device) / torch.sqrt(coeff)   # Covariance square root

    # Compute the mean for each x in the batch
    cosh_term = torch.cosh((1 - t) * beta_sqrt).to(device)
    sinh_term = torch.sinh((1 - t) * beta_sqrt).to(device)
    y_star = x / (cosh_term - sinh_term * coth1)  # Shape: (batch_size, d)

    # Now we generate multiple samples for each mean
    # Shape of z: (batch_size, num_samples, dim)
    z = torch.randn(batch_size, num_samples, d, device=device)

    # Generate new samples by broadcasting the mean and covariance
    # Shape: (batch_size, num_samples, dim)
    y = y_star.unsqueeze(1) + z @ H_sqrt  # Broadcasting mean across num_samples

    return H_inv, y_star, y


def compute_control(x, y, y_star, H_inv, beta, t, energy_function):
    """
    Computes the control vector based on the given energy function and parameters.

    This function uses the provided energy function and a custom equation involving
    softmax over sampled values of y, computes a weighted sum of y, and then calculates
    a control vector `u` for each x.

    Parameters:
    - x: Tensor of shape (batch_size, dim) representing input x.
    - y: Tensor of shape (batch_size, num_samples, dim), sampled y values for each x.
    - y_star: Tensor of shape (batch_size, dim), the mean vector for each x.
    - H_inv: Tensor of shape (dim, dim), the inverse covariance matrix.
    - beta: Scalar representing a parameter for the equation.
    - t: Scalar time factor for the equation.
    - energy_function: A function E(y) that returns the energy for each y, shape (batch_size, num_samples).

    Returns:
    - u: Tensor of shape (batch_size, dim), the computed control vector for each x.
    """

    # Ensure the code runs on the same device as x
    device = x.device

    # Hyperbolic functions of sqrt(beta) and (1-t)*sqrt(beta)
    beta_sqrt = torch.sqrt(beta).to(device)
    cosh_term = torch.cosh((1 - t) * beta_sqrt).to(device)
    sinh_term = torch.sinh((1 - t) * beta_sqrt).to(device)
    coth_beta = 1 / torch.tanh(beta_sqrt).to(device)
    print('-------------------------------------------------')
    # Step 1: Compute the energy term: -E(y)
    E_y = energy_function(y)  # Shape: (batch_size, num_samples)
    print('E_y \t', torch.mean(E_y, dim=1))


    # Term 2: Mahalanobis distance (y - y_star)^T H_inv (y - y_star)
    diff = y - y_star.unsqueeze(1)  # Shape: (batch_size, num_samples, dim)
    mahalanobis = 0.5 * torch.einsum('bnd,dd,bnd->bn', diff, H_inv.to(device), diff)  # Shape: (batch_size, num_samples)
    print('mahalanobis\t', torch.mean(mahalanobis, dim=1))


    # Term 3: Interaction term involving x and y
    x_squared = torch.sum(x.unsqueeze(1) ** 2, dim=-1, keepdim=False)  # Shape: (batch_size, 1)
    y_squared = torch.sum(y ** 2, dim=-1, keepdim=False)  # Shape: (batch_size, num_samples)
    inner_product = torch.sum(x.unsqueeze(1) * y, dim=-1)  # Shape: (batch_size, num_samples)

    interaction = -beta_sqrt * (cosh_term * (x_squared + y_squared) - 2 * inner_product) / (2 * sinh_term)
    print('interaction \t',torch.mean(interaction, dim=1))


    # Term 4: Quadratic penalty term for y
    quadratic_penalty = -0.5 * y_squared * beta_sqrt * coth_beta  # Shape: (batch_size, num_samples)
    print('quadratic_penalty\t', torch.mean(quadratic_penalty, dim=1))
  
  
    # Step 5: Combine all terms
    exponent = -E_y + mahalanobis + interaction + quadratic_penalty  # Shape: (batch_size, num_samples)
    print('exponent\t', torch.mean(exponent, dim=1))

    # Step 6: Apply softmax over the num_samples dimension
    softmax_result = torch.softmax(exponent, dim=1)  # Shape: (batch_size, num_samples)

    # Step 7: Compute the weighted sum of y using the softmax result
    weighted_y = softmax_result.unsqueeze(-1) * y  # Shape: (batch_size, num_samples, dim)
    weighted_sum = weighted_y.sum(dim=1)  # Shape: (batch_size, dim)

    # Step 8: Compute the control vector u
    u = (weighted_sum - x * cosh_term ) * (beta_sqrt / sinh_term)

    return weighted_sum, u


def energy_function(y):
    # `gmm.energy` expects input of shape (batch_size * num_samples, dim),
    # so we need to reshape y accordingly.
    batch_size, num_samples, dim = y.shape
    y_flat = y.view(-1, dim)  # Flatten y into shape (batch_size * num_samples, dim)

    # Compute the energy using the GMM energy function
    energy = gmm09.energy(y_flat)  # Shape: (batch_size * num_samples)

    # Reshape back to (batch_size, num_samples)
    return energy.view(batch_size, num_samples)

###############################################################################

eps = 1e-3
num_steps = 1001
tt = torch.linspace(eps, 1-eps, num_steps).to(device)
delta_t = tt[1] - tt[0]
beta = torch.tensor(0.001, device=device)  # Example scalar for beta
batch_size, num_samples, dim = 128, 1000, 2

x0 = torch.zeros(batch_size, dim)
x = torch.zeros(batch_size, dim)
for i, t in enumerate(tt):

    H_inv, y_star, y = normal_is(x0, beta, t, num_samples)
    w, u = compute_control(x0, y, y_star, H_inv, beta, t, energy_function)
    x = x0 +  u * delta_t + torch.sqrt(delta_t) * torch.randn_like(u).to(device)
    x0 = x
    print(i, end=' ', flush=True)

    if i % 100 == 0:  
       plt.plot(x0[:,0].cpu().numpy(), x0[:,1].cpu().numpy(), 'o', color='blue')
       plt.xlim([-10, 10])
       plt.ylim([-10, 10])
       plt.savefig(f'figs/{i}.png')
       
       
