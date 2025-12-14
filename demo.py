import torch
import numpy as np
import matplotlib.pyplot as plt

from sampler.sampler import Langevin, AdamLangevin, CyclicalAdamLangevin, CyclicalLangevin

class GaussianMixture:
    """
    A simple 2D multimodal distribution to test the sampler.
    Has two peaks (modes) at (-4, -4) and (4, 4).
    """
    def __init__(self):
        # Two modes: Mean 1 and Mean 2
        self.locs = torch.tensor([[-4.0, -4.0], [4.0, 4.0]])
        self.scale = 0.25
        
    def energy(self, x):
        """
        Calculates Potential Energy U(x) = -Log(Posterior).
        For a mixture of Gaussians, U(x) = -log( sum( exp(-0.5 * (x-mu)^2) ) )
        """
        # Calculate squared distance to each mode
        # x shape: [Batch, 2], locs shape: [2, 2]
        x = x.unsqueeze(1) # [Batch, 1, 2]
        locs = self.locs.unsqueeze(0) # [1, 2, 2]
        
        # Mahalanobis distance (simplified for isotropic covariance)
        dists = torch.sum((x - locs)**2, dim=2) / (2 * self.scale**2) # [Batch, 2]
        
        # Log-Sum-Exp trick for numerical stability
        # U(x) = -log( sum(exp(-dists)) )
        # We assume equal weights for both modes
        log_prob = torch.logsumexp(-dists, dim=1)
        
        return log_prob.sum() # Return scalar energy
    
    def score(self, x):
        x_ = x.detach().requires_grad_(True)
        log_prob = self.energy(x_)
        grad = torch.autograd.grad(log_prob.sum(), x_)[0]
        return grad


if __name__ == "__main__":
    gmm = GaussianMixture()
    true_particles = 0.25 * torch.randn(2000, 2)
    true_particles[:1000, :] -= 4.0
    true_particles[1000:, :] += 4.0

    initial_particles = torch.zeros(2000, 2, requires_grad=False)

    # ----------------------------
    # Langevin 
    print("langevin")
    lmc = Langevin(
        score_fn=gmm.score, 
        initial_particles=initial_particles,
        step_size=0.1,
        num_steps=200
    )
    particles_lmc = lmc.sample().detach().numpy()
    # Adam Langevin 
    print("adam langevin")
    adam_lmc = AdamLangevin(
        score_fn=gmm.score, 
        initial_particles=initial_particles,
        step_size=0.1,
        num_steps=200
    )
    particles_adam_lmc = adam_lmc.sample().detach().numpy()

    print("Plotting results...")
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    axs[0].scatter(true_particles[:, 0], true_particles[:, 1], s=20, alpha=0.5, color='blue')
    axs[0].set_title(f"ground truth")
    axs[0].set_xlim(-8, 8); 
    axs[0].set_ylim(-8, 8)
    axs[0].grid(True, alpha=0.3)
    axs[1].scatter(particles_lmc[:, 0], particles_lmc[:, 1], s=20, alpha=0.5, color='blue')
    axs[1].set_title(f"LMC")
    axs[1].set_xlim(-8, 8); 
    axs[1].set_ylim(-8, 8)
    axs[1].grid(True, alpha=0.3)
    axs[2].scatter(particles_adam_lmc[:, 0], particles_adam_lmc[:, 1], s=20, alpha=0.5, color='blue')
    axs[2].set_title(f"Adam LMC")
    axs[2].set_xlim(-8, 8); 
    axs[2].set_ylim(-8, 8)
    axs[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("sampling.pdf", dpi=1200)

    # ----------------------------
    print("cyclincal langevin")
    cyclical_lmc = CyclicalLangevin(
        score_fn=gmm.score,
        initial_particles=initial_particles,
        num_cycles=5,
        cycle_length=2000,
        max_step_size=0.1,
        min_step_size=1.0e-3,
        ratio=0.9
    )
    all_samples_lmc, collected_samples_lmc = cyclical_lmc.sample()

    print("cyclincal adam langevin")
    cyclical_adam_lmc = CyclicalAdamLangevin(
        score_fn=gmm.score,
        initial_particles=initial_particles,
        num_cycles=5,
        cycle_length=2000,
        max_step_size=0.1,
        min_step_size=1.0e-5,
        ratio=0.9
    )
    all_samples_adam_lmc, collected_samples_adam_lmc = cyclical_adam_lmc.sample()

    print("Plotting results...")
    all_samples_lmc = np.array(all_samples_lmc)
    collected_samples_lmc = np.array(collected_samples_lmc)
    all_samples_adam_lmc = np.array(all_samples_adam_lmc)
    collected_samples_adam_lmc = np.array(collected_samples_adam_lmc)
    
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))

    axs[0, 0].scatter(true_particles[:, 0], true_particles[:, 1], s=20, alpha=0.5, color='blue')
    axs[0, 0].set_title(f"ground truth")
    axs[0, 0].set_xlim(-8, 8); 
    axs[0, 0].set_ylim(-8, 8)
    axs[0, 0].grid(True, alpha=0.3)
    # Full Trajectory (Exploration)
    # Shows the path moving between modes
    axs[0, 1].scatter(all_samples_lmc[:, 0], all_samples_lmc[:, 1], s=20, alpha=0.5, color='blue')
    axs[0, 1].set_title(f"Full Trajectory")
    axs[0, 1].set_xlim(-8, 8); 
    axs[0, 1].set_ylim(-8, 8)
    axs[0, 1].grid(True, alpha=0.3)
    # Collected Samples (Cold)
    # Shows that we successfully sampled BOTH modes
    axs[0, 2].scatter(collected_samples_lmc[:, 0], collected_samples_lmc[:, 1], s=20, alpha=0.5, color='blue')
    axs[0, 2].set_title("Collected Samples (Low Temp Only)")
    axs[0, 2].set_xlim(-8, 8); 
    axs[0, 2].set_ylim(-8, 8)
    axs[0, 2].grid(True, alpha=0.3)

    axs[1, 0].scatter(true_particles[:, 0], true_particles[:, 1], s=20, alpha=0.5, color='blue')
    axs[1, 0].set_title(f"ground truth")
    axs[1, 0].set_xlim(-8, 8); 
    axs[1, 0].set_ylim(-8, 8)
    axs[1, 0].grid(True, alpha=0.3)
    # Full Trajectory (Exploration)
    # Shows the path moving between modes
    axs[1, 1].scatter(all_samples_adam_lmc[:, 0], all_samples_adam_lmc[:, 1], s=20, alpha=0.5, color='blue')
    axs[1, 1].set_title(f"Full Trajectory")
    axs[1, 1].set_xlim(-8, 8); 
    axs[1, 1].set_ylim(-8, 8)
    axs[1, 1].grid(True, alpha=0.3)
    # Collected Samples (Cold)
    # Shows that we successfully sampled BOTH modes
    axs[1, 2].scatter(collected_samples_adam_lmc[:, 0], collected_samples_adam_lmc[:, 1], s=20, alpha=0.5, color='blue')
    axs[1, 2].set_title("Collected Samples (Low Temp Only)")
    axs[1, 2].set_xlim(-8, 8); 
    axs[1, 2].set_ylim(-8, 8)
    axs[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("cyclincal_sampling.pdf", dpi=1200)

