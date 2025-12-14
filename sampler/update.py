from typing import Any, cast, Tuple, Dict
import math

import torch 


class SamplerUpdate:
    def __init__(
            self, 
            particles: torch.Tensor,
            defaults: dict[str, Any]
        ) -> None:
        self.particles: torch.Tensor = particles
        self.num_chains = particles.size(0)
        self.defaults = defaults

    def step(self, step_size: float):
        raise NotImplementedError
    

class LangevinUpdate(SamplerUpdate):
    def __init__(
            self, 
            particles: torch.Tensor, 
        ) -> None:
        # Define defaults
        defaults = dict()
        
        # Initialize the Base Class
        super(LangevinUpdate, self).__init__(particles, defaults)

    @torch.no_grad()
    def step(self, step_size: float) -> torch.Tensor:
        # Drift term 
        drift = cast(torch.Tensor, self.particles.grad)
        self.particles.add_(drift, alpha=step_size)

        # Diffusion term 
        noise_std = math.sqrt(2.0 * step_size)
        noise = torch.randn_like(self.particles) * noise_std
        self.particles.add_(noise)

        return self.particles


class AdamLangevinUpdate(SamplerUpdate):
    def __init__(
            self, 
            particles: torch.Tensor, 
            beta: float = 0.999, 
            weight_decay: float = 0.0,
            eps: float = 1.0e-3,
        ) -> None:
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight decay: {weight_decay}")
        if eps < 0.0:
            raise ValueError(f"Invalid eps: {eps}")
        
        # Define defaults
        defaults = dict(beta=beta, weight_decay=weight_decay, eps=eps)
        
        # Initialize the Base Class
        super(AdamLangevinUpdate, self).__init__(particles, defaults)

        # Initialize state
        self.state: Dict[str, Any] = {}

    @torch.no_grad()
    def step(self, step_size: float) -> torch.Tensor:
        beta = self.defaults["beta"]
        weight_decay = self.defaults["weight_decay"]
        eps = self.defaults["eps"]

        # Get gradient
        grad = cast(torch.Tensor, self.particles.grad)

        # Add weight decay (Gaussian prior) if requested
        if weight_decay != 0:
            grad = grad.add(self.particles, alpha=weight_decay)

        # Initialize state if first step
        if len(self.state) == 0:
            # Step index
            self.state["step"] = 0
            # Variance
            self.state["v"] = torch.zeros_like(self.particles) 

        # Get momentum and variance 
        v = self.state["v"]
        self.state["step"] += 1

        # Update variance (second moment)
        # v_t = beta * v_{t-1} + (1 - beta) * g^2
        v.mul_(beta).addcmul_(grad, grad, value=1 - beta)

        # Compute preconditioner
        preconditioner = 1.0 / (torch.sqrt(v) + eps)

        # Drift term 
        drift = preconditioner * grad
        self.particles.add_(drift, alpha=step_size)

        # Diffusion term 
        noise_std = torch.sqrt(2.0 * step_size * preconditioner)
        noise = torch.randn_like(self.particles) * noise_std
        self.particles.add_(noise)

        return self.particles


