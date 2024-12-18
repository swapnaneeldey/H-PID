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
    cosh_term = torch.cosh((1-t) * sqrt_beta)
    sinh_term = torch.sinh((1-t) * sqrt_beta)
    coth_term = 1 / torch.tanh(sqrt_beta)
    
  
    mean =  x0 / (cosh_term - sinh_term * coth_term)

    

    cov_inv = torch.zeros((d, d), dtype=x0.dtype, device=x0.device)
    
    for i in range(d):
        for j in range(d):
            if i == j:
                # Diagonal terms that evolve with time
                cov_inv[i, j] = sqrt_beta * (
                    1 / torch.tanh((1-t) * sqrt_beta) - 
                    1 / torch.tanh(sqrt_beta)
                )

    
    cov = torch.inverse(cov_inv) 
    
    # Generate samples 
    samples_list = []
    for mean_batch in mean:
        # Sample batch_size_y samples from this mean
        batch_samples = torch.distributions.MultivariateNormal(mean_batch, cov).sample((batch_size_y,))
        samples_list.append(batch_samples)
    
    # Concatenate samples along the first dimension
    samples = torch.cat(samples_list, dim=0)
    
    return cov_inv, mean, samples



def energy_function(y):  

    return gmm09.energy(y) 



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
    mahalanobis = 0.5 * torch.einsum('bnd,dd,bnd->bn', diff, H_inv.to(device), diff)  # Shape: (batch_size_x, batch_size_y)
    #print('mahalanobis\t', mahalanobis.shape) 

    # Step 1: Compute the energy term: -E(y)
    E_y = energy_function(y)  # Shape: (batch_size_y)

    # Step 2: Compute squared terms
    x_squared = torch.sum(x ** 2, dim=-1, keepdim=True)  # Shape: (batch_size_x, 1)
    y_squared = torch.sum(y ** 2, dim=-1)  # Shape: (batch_size_y)

    # Step 3: Compute inner product
    x_expanded = x.unsqueeze(1)  # Shape: (batch_size_x, 1, dim)
    y_expanded = y.unsqueeze(0)  # Shape: (1, batch_size_y, dim)
    inner_product = torch.sum(x_expanded * y_expanded, dim=-1)  # Shape: (batch_size_x, batch_size_y)

    # Step 4: Compute log(G-(t;x;y) / G+(1;y;0))
    log_G_ratio = -beta_sqrt * (cosh_term * (x_squared + y_squared.unsqueeze(0)) - 2 * inner_product) / (2 * sinh_term) + \
                  (y_squared.unsqueeze(0) * beta_sqrt / 2) * coth_beta

    # Step 5: Combine terms for the exponent
    exponent = -E_y.unsqueeze(0) + log_G_ratio - mahalanobis # Shape: (batch_size_x, batch_size_y)

    # Step 6: Apply softmax over the batch_size_y dimension
    w = torch.softmax(exponent, dim=1)  # Shape: (batch_size_x, batch_size_y)

    # Step 7: Compute the weighted sum of y
    weighted_sum = torch.sum(w.unsqueeze(-1) * y.unsqueeze(0), dim=1)  # Shape: (batch_size_x, dim)

    # Step 8: Compute the control vector u
    u = (weighted_sum - x * cosh_term) * (beta_sqrt / sinh_term)  # Shape: (batch_size_x, dim)

    return u

eps = 1e-3
num_steps = 201
tt = torch.linspace(eps, 1-eps, num_steps).to(device)
delta_t = tt[1] - tt[0]
beta = torch.tensor(1, device=device)  # Example scalar for beta
batch_size, num_samples, dim = 1000, 10, 2

x0 = torch.zeros(batch_size, dim)
x = torch.zeros(batch_size, dim)
for i, t in enumerate(tt):

    H_inv, y_star, y = generate_custom_gaussian_samples(x0, beta, t, num_samples)
    u = compute_control(x0, y, y_star, H_inv, beta, t, energy_function)
    x = x0 +  u * delta_t + torch.sqrt(delta_t) * torch.randn_like(u).to(device)
    x0 = x
    #print(i, end=' ', flush=True)

    if i % 10 == 0: 
       plt.figure()
       plt.plot(x0[:,0].cpu().numpy(), x0[:,1].cpu().numpy(), 'o', color='blue')
       #plt.plot(y[:,0].cpu().numpy(), y[:,1].cpu().numpy(), 'o', color='red')
       
       scale = 10
       plt.xlim([-scale, scale])
       plt.ylim([-scale, scale])
       plt.savefig(f'figs/{i}.png')