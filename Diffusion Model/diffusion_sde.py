import abc
import torch
import math


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