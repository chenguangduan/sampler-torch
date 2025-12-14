import torch
import numpy as np
import matplotlib.pyplot as plt
import math

from sampler.sampler import Langevin, AdamLangevin, CyclicalAdamLangevin, CyclicalLangevin

def log_prob_mog(y, means, covs, weights):
    """Compute the log-likelihood of a mixture of Gaussians

    Args:
            y (torch.Tensor of shape (batch_size, dim)): Samples to evaluate
            mean (torch.Tensor of shape (n_modes, dim)): Means of each Gaussian
            covs (torch.Tensor of shape (n_modes, dim, dim)): Covariance of each Gaussian
            weights (torch.Tensor of shape (n_modes,)): Weight of each Gaussian

    Returns:
            log_prob (torch.Tensor of shape (batch_size,)): Log-likelihood of the samples
    """

    # Compute the log_prob
    diff = y.unsqueeze(1) - means.unsqueeze(0)
    log_prob = -torch.matmul(diff.unsqueeze(-2), torch.linalg.solve(covs.unsqueeze(0), diff.unsqueeze(-1)))
    log_prob = log_prob.squeeze(-1).squeeze(-1)
    log_prob -= (y.shape[-1] * math.log(2. * math.pi) + torch.logdet(covs))
    log_prob = 0.5 * log_prob.squeeze(0)
    # Compute the prob
    log_prob += torch.log(weights / weights.sum())
    return torch.logsumexp(log_prob, dim=-1)

class MoG:
    """Distribution of a mixture of Gaussians"""

    def __init__(self, means, covs, weights):
        """Constructor

        Args:
                mean (torch.Tensor of shape (n_modes, dim)): Means of each Gaussian
                covs (torch.Tensor of shape (n_modes, dim, dim)): Covariance of each Gaussian
                weights (torch.Tensor of shape (n_modes,)): Weight of each Gaussian
        """

        self.means = means
        self.covs = covs
        self.weights = weights
        self.covariance_matrices_eye = torch.stack(
            [torch.eye(self.means.shape[-1], device=self.means.device)] * self.weights.shape[0])
        mix = torch.distributions.Categorical(weights)
        comp = torch.distributions.MultivariateNormal(loc=means, covariance_matrix=covs)
        self.dist = torch.distributions.MixtureSameFamily(mix, comp, validate_args=False)

    def sample(self, sample_shape):
        """Sample the distribution

        Args:
                sample_shape (tuple of int): Desired shape for the samples

        Returns:
                samples (torch.Tensor of shape (*sample_shape, dim))
        """

        return self.dist.sample(sample_shape)

    def log_prob(self, values):
        """Evaluate the log-likelihood of the distribution

        Args:
                values (torch.Tensor of shape (*sample_shape, dim)): Samples to evaluate

        Returns:
                log_prob (torch.Tensor of shape sample_shape): Log-likelihood of the samples
        """

        return self.dist.log_prob(values)

    def log_prob_p_t(self, y, t, sigma, alpha):
        """Compute the likelihood of the noised version of the distribution
                Y = alpha(t) * X + sigma * sqrt(t) * Z  with X ~ dist and Z ~ N(0,I)
        Args:
                y (torch.Tensor of shape (batch_size, dim)): Samples to evaluate
                t (float): Current time
                sigma (float): The current noise level
                alpha (Alpha): Object containing alpha details
        Returns:
                log_prob (torch.Tensor of shape (batch_size,)): Log-likelihood of the samples
        """

        # Compute the covariances
        covs = (alpha.alpha(t)**2) * self.covs.clone() + (torch.square(sigma) * t) * self.covariance_matrices_eye
        # Compute the log_prob
        return log_prob_mog(y, alpha.alpha(t) * self.means, covs, self.weights)

    def mnm_sigma(self):
        """Compute the value of sigma based on Eq (4.3) of "Chain of Log-Concave Markov Chains" (arXiv:2305.19473)

        Returns:
            sigma (float): Value of sigma
        """

        R = float(torch.cdist(self.means.unsqueeze(0), self.means.unsqueeze(0)).max())
        tau = float(torch.sqrt(self.covs.max()))
        return math.sqrt(max(0.0, R**2 - tau**2))


class Grid8x8Mixture(MoG):
    """Mixture of 16 Gaussians arranged in a 4×4 grid"""

    def __init__(self, device, grid_size=8, spacing=10.0, scale=0.1):
        """Constructor

        Args:
                device (torch.device): Device to use for computations
                grid_size (int): Size of the grid (default is 4, resulting in 4×4=16 Gaussians)
                spacing (float): Spacing between grid points (default is 10.0)
                scale (float): Scale of the individual Gaussians (default is 0.2)
        """
        n_gaussians = grid_size * grid_size
        means = []
        
        # Create a 4×4 grid of means
        # Center the grid around origin
        offset = (grid_size - 1) * spacing / 2.0
        for i in range(grid_size):
            for j in range(grid_size):
                x = i * spacing - offset
                y = j * spacing - offset
                means.append(torch.Tensor([x, y]))
        
        means = torch.stack(means).to(device)
        covs = scale * torch.stack([torch.eye(2, device=device)] * n_gaussians)
        weights = (1 / n_gaussians) * torch.ones((n_gaussians,), device=device)
        super().__init__(means, covs, weights)
    
    def score(self, x):
        x_ = x.detach().requires_grad_(True)
        log_prob = self.log_prob(x_)
        grad = torch.autograd.grad(log_prob.sum(), x_)[0]
        return grad


if __name__ == "__main__":
    gmm = Grid8x8Mixture(device="cpu")
    true_particles = gmm.sample((2000, 2))

    initial_particles = torch.zeros_like(true_particles)
    # ----------------------------
    # Langevin 
    print("langevin")
    lmc = Langevin(
        score_fn=gmm.score, 
        initial_particles=initial_particles,
        step_size=0.1,
        num_steps=100
    )
    particles_lmc = lmc.sample().detach().numpy()
    # Adam Langevin 
    print("adam langevin")
    adam_lmc = AdamLangevin(
        score_fn=gmm.score, 
        initial_particles=initial_particles,
        step_size=0.1,
        num_steps=100
    )
    particles_adam_lmc = adam_lmc.sample().detach().numpy()

    print("Plotting results...")
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    axs[0].scatter(true_particles[:, 0], true_particles[:, 1], s=20, alpha=0.5, color='blue')
    axs[0].set_title(f"ground truth")
    axs[0].set_xlim(-40,40)
    axs[0].set_ylim(-40,40)
    axs[0].grid(True, alpha=0.3)
    axs[1].scatter(particles_lmc[:, 0], particles_lmc[:, 1], s=20, alpha=0.5, color='blue')
    axs[1].set_title(f"LMC")
    axs[1].set_xlim(-40,40)
    axs[1].set_ylim(-40,40)
    axs[1].grid(True, alpha=0.3)
    axs[2].scatter(particles_adam_lmc[:, 0], particles_adam_lmc[:, 1], s=20, alpha=0.5, color='blue')
    axs[2].set_title(f"Adam LMC")
    axs[2].set_xlim(-40,40)
    axs[2].set_ylim(-40,40)
    axs[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("sampling_mog_8x8.pdf", dpi=1200)

    