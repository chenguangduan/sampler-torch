from typing import Any, cast, Tuple, Callable, List
import math

import torch 

from .update import SamplerUpdate, LangevinUpdate, AdamLangevinUpdate 


class Sampler:
    def __init__(
            self, 
            score_fn: Callable[[torch.Tensor], torch.Tensor],
            initial_particles: torch.Tensor,
            step_size: float, 
            num_steps: int, 
        ) -> None:
        self.score_fn = score_fn
        self.particles = initial_particles.clone().requires_grad_(True)
        self.step_size = step_size
        self.num_steps = num_steps
        self.update: SamplerUpdate

    def compute_score(self) -> None:
        score = self.score_fn(self.particles)
        score = torch.clamp(score, min=-1.0e2, max=1.0e2)
        self.particles.grad = score
    
    def sample(self) -> torch.Tensor:
        for _ in range(self.num_steps):
            self.compute_score()
            self.update.step(step_size=self.step_size)
        return self.particles
    

class Langevin(Sampler):
    def __init__(            
            self, 
            score_fn: Callable[[torch.Tensor], torch.Tensor],
            initial_particles: torch.Tensor,
            step_size: float,
            num_steps: int,
        ) -> None:
        super(Langevin, self).__init__(
            score_fn=score_fn, 
            initial_particles=initial_particles,
            step_size=step_size, 
            num_steps=num_steps, 
        )
        self.update = LangevinUpdate(
            particles=self.particles
        )
    

class AdamLangevin(Sampler):
    def __init__(            
            self, 
            score_fn: Callable[[torch.Tensor], torch.Tensor],
            initial_particles: torch.Tensor,
            step_size: float,
            num_steps: int,
        ) -> None:
        super(AdamLangevin, self).__init__(
            score_fn=score_fn, 
            initial_particles=initial_particles, 
            step_size=step_size,
            num_steps=num_steps,
        )
        self.update = AdamLangevinUpdate(
            particles=self.particles
        )
    

class CyclicalSampler:
    def __init__(            
            self, 
            score_fn: Callable[[torch.Tensor], torch.Tensor],
            initial_particles: torch.Tensor,
            num_cycles: int,
            cycle_length: int,
            max_step_size: float,
            min_step_size: float,
            ratio: float = 0.95,
        ) -> None:
        self.score_fn = score_fn
        self.particles = initial_particles.clone().requires_grad_(True)
        self.num_cycles = num_cycles 
        self.cycle_length = cycle_length 
        self.max_step_size = max_step_size
        self.min_step_size = min_step_size
        self.ratio=ratio
        self.num_steps = num_cycles * cycle_length
        self.update: SamplerUpdate

    def compute_score(self) -> None:
        score = self.score_fn(self.particles)
        score = torch.clamp(score, min=-1.0e2, max=1.0e2)
        self.particles.grad = score

    def get_cycle_progress(
            self,
            step: int
        ) -> float:
        cycle_length = self.cycle_length
        cycle_progress = (step % cycle_length) / cycle_length
        return cycle_progress

    def get_cyclical_step_size(
            self,
            step: int
        ) -> float:
        cycle_progress = self.get_cycle_progress(step)
        # Cosine annealing from max_lr down to min_lr
        max_step_size = self.max_step_size
        min_step_size = self.min_step_size
        cos_out = 0.5 * (1 + math.cos(math.pi * cycle_progress))
        step_size = min_step_size + (max_step_size - min_step_size) * cos_out
        return step_size
    
    def sample(self) -> Tuple[List, List]:
        all_particles = []
        collected_particles = []
        for idx_step in range(self.num_steps):
            self.compute_score()
            step_size = self.get_cyclical_step_size(idx_step)
            self.update.step(step_size=step_size)
            cycle_progress = self.get_cycle_progress(idx_step)

            current_particles = self.particles.detach().numpy().copy()
            all_particles.append(current_particles)
            if cycle_progress > self.ratio:
                collected_particles.append(current_particles)

        return all_particles, collected_particles


class CyclicalLangevin(CyclicalSampler):
    def __init__(            
            self, 
            score_fn: Callable[[torch.Tensor], torch.Tensor],
            initial_particles: torch.Tensor,
            num_cycles: int,
            cycle_length: int,
            max_step_size: float,
            min_step_size: float,
            ratio: float = 0.95,
        ) -> None:
        super(CyclicalLangevin, self).__init__(
            score_fn=score_fn,
            initial_particles=initial_particles,
            num_cycles=num_cycles,
            cycle_length=cycle_length,
            max_step_size=max_step_size,
            min_step_size=min_step_size,
            ratio=ratio
        )
        self.update = LangevinUpdate(
            particles=self.particles
        )


class CyclicalAdamLangevin(CyclicalSampler):
    def __init__(            
            self, 
            score_fn: Callable[[torch.Tensor], torch.Tensor],
            initial_particles: torch.Tensor,
            num_cycles: int,
            cycle_length: int,
            max_step_size: float,
            min_step_size: float,
            ratio: float = 0.95,
        ) -> None:
        super(CyclicalAdamLangevin, self).__init__(
            score_fn=score_fn,
            initial_particles=initial_particles,
            num_cycles=num_cycles,
            cycle_length=cycle_length,
            max_step_size=max_step_size,
            min_step_size=min_step_size,
            ratio=ratio
        )
        self.update = AdamLangevinUpdate(
            particles=self.particles
        )

