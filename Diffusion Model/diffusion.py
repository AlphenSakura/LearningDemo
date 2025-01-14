import math
import time 
import numpy as np
import torch
import torch.nn as nn
import torch.functional as F


class WeightedLoss():
    def __init__(self):
        super(WeightedLoss, self).__init__()
    
    def forward(self, pred, targ, weights = 1.0):
        loss = self._loss(pred, targ)
        weightedLoss = (loss * weights).mean()
        return weightedLoss

# L1范数
class WeightedL1Loss(WeightedLoss):
    def _loss(self, pred, targ):
        return torch.abs(pred - targ)

# L2范数    
class WeightedL2Loss(WeightedLoss):
    def _loss(self, pred, targ):
        return F.mse_loss(pred, targ, reduction='none')

Losses = {
    'l1': WeightedL1Loss,
    'l2': WeightedL2Loss,
}


# Sinusoidal Positional Embedding
# 位置编码
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super(SinusoidalPosEmb, self).__init__()
        self.dim = dim
        
    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb

class MLP(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, device, t_dim):
        super(MLP, self).__init__()
        
        
        self.action_dim = action_dim
        self.device = device
        self.t_dim = t_dim
        
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim,t_dim*2),
            nn.Mish(),
            nn.Linear(t_dim*2,t_dim)
        )
        
        input_dim = state_dim + action_dim + t_dim
        self.mid_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish()
        )
        self.final_layer = nn.Linear(hidden_dim, action_dim)
        
        self.init_weights()
        
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
    
    
    def forward(self, x, time, state):
        t_emb = self.time_mlp(time)
        x = torch.cat([x, state, t_emb], dim=1)
        x = self.mid_layer(x)
        return self.final_layer(x)
 
 
class Diffusion(nn.Module):
    def __init__(self, loss_type, beta_schedule = 'linear', clip_denoised = True, predict_epsilon = True, **kwargs):
        super(Diffusion, self).__init__()       
        self.state_dim = kwargs['state_dim']# 没有传入该参数会报错
        self.action_dim = kwargs['action_dim']
        self.hidden_dim = kwargs['hidden_dim']
        self.T = kwargs['T']
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon
        self.device = torch.device(kwargs['device'])
        
        
        if beta_schedule == 'linear':
            betas = torch.linspace(1e-3, 2e-2, self.T, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, 0)   # [1, 2, 3] ->` [1, 1*2, 1*2*3]
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]]) # [1, 2, 3] -> [1, 1, 1*2]
        
        self.register_buffer('betas', betas)    # register_buffer()方法可以将tensor注册为模型的属性, 但是不会被更新, 加快
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        
        #前向过程
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        
        #反向过程
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer('sqrt_recipm_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod - 1))
        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2', (1.0 - alphas_cumprod_prev) * torch.sqrt(betas) / (1.0 - alphas_cumprod))

        self.loss = Losses[loss_type]()

        def forward(self, state, *args, **kwargs):
            return self.sample(state, *args, **kwargs) 
if __name__ == '__main__':