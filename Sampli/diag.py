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
gmm09 = NineGaussianMixture(device=device, dim=2)



num_samples = 1000
y = gmm09.sample(num_samples)
energies =  gmm09.energy(y)
log_Z = torch.logsumexp(-energies, dim=0) - torch.log(torch.tensor(num_samples, dtype=torch.float32))

print(log_Z) 




