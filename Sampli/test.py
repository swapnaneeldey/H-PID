# Assuming all your previous imports and class definitions are here
import abc
import numpy as np
import random 
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
import torch.distributions as D
from torch.distributions.mixture_same_family import MixtureSameFamily

device = 'cpu'
scale = 10

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
    def __init__(self, device, scale=0.5, dim=2):
        super().__init__()
        self.device = device
        self.data = torch.tensor([0.0])
        self.data_ndim = 2

        mean_ls = [
            [-5., -5.], [-5., 0.], [-5., 5.],
            [0., -5.], [0., 0.], [0., 5.],
            [5., -5.], [5., 0.], [5., 5.],
        ]

        mean_ls = [[x*1 for x in sublist] for sublist in mean_ls]
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

def energy_function(y):

    return gmm09.energy(y)



def compute_control(x, y, beta, t, energy_function):
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
    exponent = -E_y.unsqueeze(0) + log_G_ratio  # Shape: (batch_size_x, batch_size_y)

    # Step 6: Apply softmax over the batch_size_y dimension
    w = torch.softmax(exponent, dim=1)  # Shape: (batch_size_x, batch_size_y)

    # Step 7: Compute the weighted sum of y
    weighted_sum = torch.sum(w.unsqueeze(-1) * y.unsqueeze(0), dim=1)  # Shape: (batch_size_x, dim)

    # Step 8: Compute the control vector u
    u = (weighted_sum - x * cosh_term) * (beta_sqrt / sinh_term)  # Shape: (batch_size_x, dim)

    return u

#########################################################

# 1. Verify target distribution
def plot_target_distribution(gmm, num_samples=1024, filename="target_distribution.png"):
    samples = gmm.sample(num_samples)
    plt.figure(figsize=(10, 10))
    plt.scatter(samples[:,0].cpu().numpy(), samples[:,1].cpu().numpy(), alpha=0.1)
    plt.title("Target Distribution")
    plt.xlim([-scale, scale])
    plt.ylim([-scale, scale])
    plt.savefig(filename)
    plt.close()

plot_target_distribution(gmm09)

# 2. Implement Langevin dynamics sampler
def langevin_dynamics(gmm, num_steps=201, step_size=0.005, initial_samples=None):
    if initial_samples is None:
        initial_samples = torch.randn(1000, 2) * 2
    
    samples = initial_samples.clone().requires_grad_(True)
    
    for i in range(num_steps):
        noise = torch.randn_like(samples) * np.sqrt(2 * step_size)
        energy = gmm.energy(samples)
        grad = torch.autograd.grad(energy.sum(), samples)[0]
        samples.data = samples.data - step_size * grad + noise
        
        if i % 10 == 0:
            plt.figure(figsize=(10, 10))
            plt.scatter(samples[:,0].detach().cpu().numpy(), samples[:,1].detach().cpu().numpy(), alpha=0.1)
            plt.title(f"Langevin Dynamics - Step {i}")
            plt.xlim([-scale, scale])
            plt.ylim([-scale, scale])
            plt.savefig(f"Langevin/langevin_step_{i}.png")
            plt.close()
    
    return samples.detach()

langevin_samples = langevin_dynamics(gmm09)

# 3. Modify main loop with diagnostics
def compute_control_with_diagnostics(x, y, beta, t, energy_function):
    u = compute_control(x, y, beta, t, energy_function)
    
    # Compute some statistics
    u_norm = torch.norm(u, dim=1).mean().item()
    x_norm = torch.norm(x, dim=1).mean().item()
    
    return u, u_norm, x_norm

eps = 1e-3
num_steps = 201
tt = torch.linspace(eps, 1-eps, num_steps).to(device)
delta_t = tt[1] - tt[0]
beta = torch.tensor(0.0000000000000000000000001, device=device)
batch_size, dim = 1024, 2

#x0 = (2 * torch.rand(batch_size, dim) - 1) * 10
#x0 = torch.randn(batch_size, dim) * 11
x0 = torch.zeros(batch_size, dim) 

x = x0.clone()

y = gmm09.sample(1000)

diagnostics = []

for i, t in enumerate(tt):
    u, u_norm, x_norm = compute_control_with_diagnostics(x, y, beta, t, gmm09.energy)
    x_new = x + u * delta_t + torch.sqrt(delta_t) * torch.randn_like(u).to(device)
    
    # Compute change in x
    delta_x_norm = torch.norm(x_new - x, dim=1).mean().item()
    
    
    
    diagnostics.append((t.item(), u_norm, x_norm, delta_x_norm))
    
    if i % 20 == 0:
        print(f'Step {i}, t={t.item():.4f}, u_norm={u_norm:.4f}, x_norm={x_norm:.4f}, delta_x_norm={delta_x_norm:.4f}')
        plt.figure(figsize=(10, 10))
        plt.scatter(y[:,0].cpu().numpy(), y[:,1].cpu().numpy(), color='red', alpha=0.7)
        plt.scatter(x[:,0].cpu().numpy(), x[:,1].cpu().numpy(), color='blue', alpha=0.9)
        plt.xlim([-scale, scale])
        plt.ylim([-scale, scale])
        #plt.legend()
        plt.tight_layout()
        plt.xticks([])
        plt.yticks([])
        #plt.title(f'Step {i}, t={t.item():.4f}')
        plt.savefig(f'figs/step_{i:04d}.png')
        plt.close()

    x = x_new



plt.figure(figsize=(10, 10))
plt.scatter(y[:,0].cpu().numpy(), y[:,1].cpu().numpy(), color='red', alpha=0.7)
plt.scatter(x[:,0].cpu().numpy(), x[:,1].cpu().numpy(), color='blue', alpha=0.9)
plt.xlim([-scale, scale])
plt.ylim([-scale, scale])
#plt.legend()
plt.tight_layout()
plt.xticks([])
plt.yticks([])
#plt.title(f'Final Step {i}, t={t.item():.4f}')
plt.savefig(f'figs/step_0200.png')
plt.close()

# Plot diagnostics
diagnostics = np.array(diagnostics)
plt.figure(figsize=(15, 5))
plt.plot(diagnostics[:,0], diagnostics[:,1], label='u_norm')
plt.plot(diagnostics[:,0], diagnostics[:,2], label='x_norm')
plt.plot(diagnostics[:,0], diagnostics[:,3], label='delta_x_norm')
#plt.legend()
plt.xlabel('t')
plt.ylabel('Norm')
plt.title('Diagnostics')
plt.savefig('diagnostics.png')
plt.close()


# compute partition function
num_samples = batch_size
energies =  gmm09.energy(x)
log_Z = torch.logsumexp(-energies, dim=0) - torch.log(torch.tensor(num_samples, dtype=torch.float32))
print("Estimated Samples", log_Z) 

energies =  gmm09.energy(y)
log_Z = torch.logsumexp(-energies, dim=0) - torch.log(torch.tensor(1000, dtype=torch.float32))
print("True Samples", log_Z)


"""
estimated_energies = gmm09.energy(x)
true_energies = gmm09.energy(y)


# Plot histograms
plt.figure(figsize=(10, 6))
plt.hist(estimated_energies.cpu().numpy(), bins=50, alpha=0.99, label='Estimated')
plt.hist(true_energies.cpu().numpy(), bins=50, alpha=0.2, label='True')
plt.xlabel('Energy')
plt.ylabel('Count')
plt.legend()
plt.title('Energy Distributions')
plt.savefig('E.png')
plt.close()
"""
