from typing import cast
import torch
import numpy as np
import matplotlib.pyplot as plt
import math

from sampler.sampler import Langevin, AdamLangevin, CyclicalAdamLangevin, CyclicalLangevin

class PolarTransform(torch.distributions.transforms.Transform):
    """Polar transformation"""

    domain = torch.distributions.constraints.real_vector
    codomain = torch.distributions.constraints.real_vector
    bijective = True
    sign = +1

    def __eq__(self, other):
        return isinstance(other, PolarTransform)

    def _call(self, x):
        return torch.stack([
            x[..., 0] * torch.cos(x[..., 1]),
            x[..., 0] * torch.sin(x[..., 1])
        ], dim=-1)

    def _inverse(self, y):
        x = torch.stack([
            torch.norm(y, p=2, dim=-1),
            torch.atan2(y[..., 1], y[..., 0])
        ], dim=-1)
        x[..., 1] = x[..., 1] + (x[..., 1] < 0).type_as(y) * (2 * torch.pi)
        return x

    def log_abs_det_jacobian(self, x, y):
        return torch.log(x[..., 0])

class Rings(torch.nn.Module):
    """Rings distribution"""

    def __init__(self, device, num_modes=8, radius=1.0, sigma=0.05, validate_args=False):
        """Constructor

        The distribution is centered at 0

        Args:
            device (torch.device): Device used for computations
            num_modes (int): Number of circles (default is 4)
            radius (float): Radius of the smallest circle (default is 1.0)
            sigma (float): Width of the circles (default is 0.15)
        """

        super().__init__()
        self.dummy_param = torch.nn.Parameter(torch.empty(0, device=device))
        # Make the radius distribution
        self.radius_dist = torch.distributions.MixtureSameFamily(
            mixture_distribution=torch.distributions.Categorical(
                torch.ones((num_modes,), device=device)),
            component_distribution=torch.distributions.Normal(
                loc=radius * (torch.arange(num_modes).to(device) + 1),
                scale=sigma
            )
        )
        # Make the angle distribution
        self.angle_dist = torch.distributions.Uniform(
            low=torch.zeros((1,), device=device).squeeze(),
            high=2 * torch.pi * torch.ones((1,), device=device).squeeze()
        )
        # Make the polar transform
        self.transform = PolarTransform()
        # Set the extreme values
        self.x_min = - radius * num_modes - sigma
        self.x_max = radius * num_modes + sigma
        self.y_min = - radius * num_modes - sigma
        self.y_max = radius * num_modes + sigma

    def sample(self, sample_shape=torch.Size()):
        """Sample the distribution

        Args:
            sample_shape (tuple of int): Shape of the samples

        Returns
            samples (torch.Tensor of shape (*sample_shape, 2)): Samples
        """

        r = self.radius_dist.sample(sample_shape)
        theta = self.angle_dist.sample(sample_shape)
        if len(sample_shape) == 0:
            x = torch.FloatTensor([r, theta])
        else:
            x = torch.stack([r, theta], dim=1)
        return self.transform(x)

    def log_prob(self, value):
        """Evaluate the log-likelihood of the distribution

        Args:
            value (torch.Tensor of shape (batch_size, 2)): Sample

        Returns
            log_prob (torch.Tensor of shape (batch_size,)): Log-likelihood of the samples
        """

        x = self.transform.inv(value)
        return self.radius_dist.log_prob(x[..., 0]) + self.angle_dist.log_prob(x[..., 1]
                                                                               ) - self.transform.log_abs_det_jacobian(x, value)

    def _apply(self, fn):
        """Apply the fn function on the distribution

        Args:
            fn (function): Function to apply on tensors
        """

        new_self = super(Rings, self)._apply(fn)
        # Radius distribution
        new_self.radius_dist.mixture_distribution.probs = fn(
            new_self.radius_dist.mixture_distribution.probs)
        new_self.radius_dist.component_distribution.loc = fn(
            new_self.radius_dist.component_distribution.loc)
        new_self.radius_dist.component_distribution.scale = fn(
            new_self.radius_dist.component_distribution.scale)
        # Angle distribution
        new_self.angle_dist.low = fn(new_self.angle_dist.low)
        new_self.angle_dist.high = fn(new_self.angle_dist.high)
        return new_self
    
    def score(self, x):
        x_ = x.detach().requires_grad_(True)
        log_prob = self.log_prob(x_)
        grad = torch.autograd.grad(log_prob.sum(), x_)[0]
        return grad


if __name__ == "__main__":
    gmm = Rings(device="cpu")
    true_particles = cast(torch.Tensor, gmm.sample(sample_shape=torch.Size((2000,))))

    initial_particles = torch.randn_like(true_particles)
    # ----------------------------
    # Langevin 
    print("langevin")
    lmc = Langevin(
        score_fn=gmm.score, 
        initial_particles=initial_particles,
        step_size=0.005,
        num_steps=10000
    )
    particles_lmc = lmc.sample().detach().numpy()
    # Adam Langevin 
    print("adam langevin")
    adam_lmc = AdamLangevin(
        score_fn=gmm.score, 
        initial_particles=initial_particles,
        step_size=0.05,
        num_steps=10000
    )
    particles_adam_lmc = adam_lmc.sample().detach().numpy()

    print("Plotting results...")
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    axs[0].scatter(true_particles[:, 0], true_particles[:, 1], s=20, alpha=0.5, color='blue')
    axs[0].set_title(f"ground truth")
    axs[0].set_xlim(-10,10)
    axs[0].set_ylim(-10,10)
    axs[0].grid(True, alpha=0.3)
    axs[1].scatter(particles_lmc[:, 0], particles_lmc[:, 1], s=20, alpha=0.5, color='blue')
    axs[1].set_title(f"LMC")
    axs[1].set_xlim(-10,10)
    axs[1].set_ylim(-10,10)
    axs[1].grid(True, alpha=0.3)
    axs[2].scatter(particles_adam_lmc[:, 0], particles_adam_lmc[:, 1], s=20, alpha=0.5, color='blue')
    axs[2].set_title(f"Adam LMC")
    axs[2].set_xlim(-10,10)
    axs[2].set_ylim(-10,10)
    axs[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("sampling_rings.pdf", dpi=1200)


