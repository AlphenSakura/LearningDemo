import abc
import torch
import math
import numpy as np
import torch.nn as nn
from diffusion import MLP


class SDEBase(abc.ABC):
    def __init__(self, T):
        self.T = T
        self.dt = 1 / T
        
    @abc.abstractmethod
    def drift(self, x_t, t):
        pass
        
    @abc.abstractmethod
    def dispersion(self, x_t, t):
        pass
    
    def dw(self, x):
        return torch.randn_like(x) * math.sqrt(self.dt)
    
    def reverse_ode(self, x, t, score):
        dx = (self.drift(x, t) - 0.5 * self.dispersion(x, t)**2 * score) * self.dt 
        return x - dx
    
    def reverse_sde(self, x, t, score):
        dx = (self.drift(x, t)-self.dispersion(x, t)**2 * score) * self.dt + \
            self.dispersion(x, t) * self.dw(x) *(t > 0)
        return x - dx
    
    def forward_step(self, x, t):
        dx = self.drift(x, t) * self.dt + self.dispersion(x, t) * self.dw(x)
        return x + dx
    
    def forward(self,x_0):
        x = x_0
        for t in range(self.T):
            x = self.forward_step(x, t)
        return x
    
    def reverse(self, x_T, score, mode):
        x = x_T
        for t in reversed(range(self.T)):
            score_value = score(x, t)
            if mode == 'sde':
                x = self.reverse_sde(x, t, score_value)
            elif mode == 'ode':
                x = self.reverse_ode(x, t, score_value)
        return x
    
 
    
def vp_beta_schedule(timesteps, dtype = torch.float32):
    t = np.arange(1, timesteps + 1)
    T = timesteps
    b_max = 10
    b_min = 0.1
    alpha = np.exp(-b_min / T -0.5 * (b_max - b_min) * (2 * t - 1) / T ** 2)
    betas = 1 - alpha
    return torch.tensor(betas, dtype = dtype)


class DiffusionSDE(SDEBase):
    def __init__(self, T, schedule):
        super().__init__(T=T)
        
        if schedule == 'vp':
            self.thetas = vp_beta_schedule(T)
        
        self.sigmas = torch.sqrt(2 * self.thetas)
        
        thetas_cumsum = torch.cusum(self.thetas, dim = 0)
        self.dt = -math.log(1e-3) / thetas_cumsum[-1]
        self.thetas_bar = thetas_cumsum * self.dt
        self.vars = 1 - torch.exp(-2 * self.thetas_bar)
        self.stds = torch.sqrt(self.vars)
        
                
    def drift(self, x_t, t):
        return -self.thetas[t] * x_t
    
    def dispersion(self, x_t, t):
        return self.sigmas[t]
    
    def compute_noise_score(self, noise, t):
        return -noise / self.stds(noise, t)
    
    
class DiffusionSDEPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, T, max_action):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.T = T
        self.maxaction = max_action
        self.model = MLP(state_dim, action_dim, hidden_dim )
        
        self.sde = DiffusionSDE(T)
        
    def score_fn(self, a_t, t, state):
        noise = self.model(a_t, t, state)
        return self.sde.compute_noise_score(noise, t)