import torch
import torch.nn as nn
from copy import deepcopy
from torch.utils.data import DataLoader, Dataset
import numpy as np


class PoiLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,pred,target):
       # p =clamp_probs(p)
    # Create a NegativeBinomial distribution
        poi_dist = torch.distributions.poisson.Poisson(pred)
    
    # Calculate the negative log likelihood
        log_probs = poi_dist.log_prob(target)
    
    # Return the negative mean log likelihood
        loss = -torch.mean(log_probs)
    
        return loss


class MSELoss(nn.Module):
    '''MSE loss that sums over output dimensions.'''
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        loss = torch.sum((pred - target) ** 2, dim=-1)
        return torch.mean(loss)


class Accuracy(nn.Module):
    '''0-1 classification loss.'''
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        return (torch.argmax(pred, dim=1) == target).float().mean()
