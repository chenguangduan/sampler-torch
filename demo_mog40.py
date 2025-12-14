import torch
import numpy as np
import matplotlib.pyplot as plt
import math

from sampler.sampler import Langevin, AdamLangevin, CyclicalAdamLangevin, CyclicalLangevin

class MoG40(torch.nn.Module):
    """40-component Gaussian Mixture Model
    
    This distribution consists of 40 Gaussian components with random means
    and variances, similar to the GMM in utils.py from DiGS-master.
    """

    def __init__(self, device, dim=2, n_mixes=40, loc_scaling=40.0, 
                 log_var_scaling=1.0, seed=0):
        """Constructor

        Args:
            device (torch.device): Device used for computations
            dim (int): Dimension of the problem (default is 2)
            n_mixes (int): Number of Gaussian components (default is 40)
            loc_scaling (float): Scale of the problem - determines how far apart 
                                 the modes of each Gaussian component will be (default is 40.0)
            log_var_scaling (float): Variance scaling factor for each Gaussian (default is 1.0)
            seed (int): Random seed for reproducibility (default is 0)
        """

        super().__init__()
        self.seed = seed
        self.n_mixes = n_mixes
        self.dim = dim
        self.device = device

        # Set random seed for initialization
        torch.manual_seed(seed)
        
        # Generate random means: (n_mixes, dim)
        mean = (torch.rand((n_mixes, dim)) - 0.5) * 2 * loc_scaling
        
        # Generate log variances: (n_mixes, dim)
        log_var = torch.ones((n_mixes, dim)) * log_var_scaling

        # Register buffers (these are not trainable parameters)
        self.register_buffer("cat_probs", torch.ones(n_mixes))
        self.register_buffer("locs", mean)
        # Use softplus to ensure positive scale
        self.register_buffer("scale_trils", torch.diag_embed(
            torch.nn.functional.softplus(log_var)))
        
        self.to(self.device)

    def to(self, device):
        """Move the distribution to the specified device"""
        if device == "cuda":
            if torch.cuda.is_available():
                self.cuda()
            else:
                self.cpu()
        else:
            self.cpu()
        self.device = device
        return self

    @property
    def distribution(self):
        """Get the underlying PyTorch distribution"""
        mix = torch.distributions.Categorical(self.cat_probs.to(self.device))
        com = torch.distributions.MultivariateNormal(
            self.locs.to(self.device),
            scale_tril=self.scale_trils.to(self.device),
            validate_args=False)
        return torch.distributions.MixtureSameFamily(
            mixture_distribution=mix,
            component_distribution=com,
            validate_args=False)

    def log_prob(self, x: torch.Tensor):
        """Evaluate the log-likelihood of the distribution

        Args:
            x (torch.Tensor of shape (batch_size, dim)): Samples to evaluate

        Returns:
            log_prob (torch.Tensor of shape (batch_size,)): Log-likelihood of the samples
        """
        log_prob = self.distribution.log_prob(x)
        # Mask out extremely low probabilities to avoid numerical issues
        mask = torch.zeros_like(log_prob)
        mask[log_prob < -1e4] = -torch.tensor(float("inf"))
        log_prob = log_prob + mask
        return log_prob

    def sample(self, sample_shape=(1,)):
        """Sample the distribution

        Args:
            sample_shape (tuple of int): Desired shape for the samples

        Returns:
            samples (torch.Tensor of shape (*sample_shape, dim)): Samples from the distribution
        """
        return self.distribution.sample(sample_shape)
    
    def score(self, x):
        x_ = x.detach().requires_grad_(True)
        log_prob = self.log_prob(x_)
        grad = torch.autograd.grad(log_prob.sum(), x_)[0]
        return grad


if __name__ == "__main__":
    torch.manual_seed(0)
    gmm = MoG40(
        device="cpu",
        dim=2,
        n_mixes=40,
        loc_scaling=40.0,
        log_var_scaling=1.0,
        seed=0
    )
    true_particles = gmm.sample((2000,))

    initial_particles = torch.randn_like(true_particles)
    # ----------------------------
    # Langevin 
    print("langevin")
    lmc = Langevin(
        score_fn=gmm.score, 
        initial_particles=initial_particles,
        step_size=0.5,
        num_steps=50000
    )
    particles_lmc = lmc.sample().detach().numpy()
    # Adam Langevin 
    print("adam langevin")
    adam_lmc = AdamLangevin(
        score_fn=gmm.score, 
        initial_particles=initial_particles,
        step_size=0.5,
        num_steps=50000
    )
    particles_adam_lmc = adam_lmc.sample().detach().numpy()

    print("Plotting results...")
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    axs[0].scatter(true_particles[:, 0], true_particles[:, 1], s=20, alpha=0.5, color='blue')
    axs[0].set_title(f"ground truth")
    axs[0].set_xlim(-45,45)
    axs[0].set_ylim(-45,45)
    axs[0].grid(True, alpha=0.3)
    axs[1].scatter(particles_lmc[:, 0], particles_lmc[:, 1], s=20, alpha=0.5, color='blue')
    axs[1].set_title(f"LMC")
    axs[1].set_xlim(-45,45)
    axs[1].set_ylim(-45,45)
    axs[1].grid(True, alpha=0.3)
    axs[2].scatter(particles_adam_lmc[:, 0], particles_adam_lmc[:, 1], s=20, alpha=0.5, color='blue')
    axs[2].set_title(f"Adam LMC")
    axs[2].set_xlim(-45,45)
    axs[2].set_ylim(-45,45)
    axs[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("sampling_mog40.pdf", dpi=1200)

    
