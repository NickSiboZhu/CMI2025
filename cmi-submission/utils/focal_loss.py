#!/usr/bin/env python3
"""
Focal Loss implementation for gesture recognition with class imbalance.

This implementation is designed specifically for the CMI gesture recognition task
where BFRB gestures (minority classes) need extra attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in gesture recognition.
    
    Focal Loss: FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    Args:
        alpha: Balancing factor for rare class (default: 1.0)
        gamma: Focusing parameter (default: 2.0) 
        weight: Class weights (optional)
        ignore_index: Index to ignore (default: -100)
        size_average: Whether to average (deprecated, use reduction)
        reduce: Whether to reduce (deprecated, use reduction)
        reduction: Specifies the reduction to apply ('none' | 'mean' | 'sum')
    """
    
    def __init__(self, alpha=1.0, gamma=2.0, weight=None, ignore_index=-100, 
                 size_average=None, reduce=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction
        
        # Handle deprecated parameters
        if size_average is not None or reduce is not None:
            self.reduction = 'mean' if size_average else 'sum'
    
    def forward(self, input, target):
        """
        Forward pass of Focal Loss.
        
        Args:
            input: Predicted logits (batch_size, num_classes)
            target: Ground truth labels (batch_size,)
            
        Returns:
            Focal loss value
        """
        # Compute cross entropy
        ce_loss = F.cross_entropy(input, target, weight=self.weight, 
                                  ignore_index=self.ignore_index, reduction='none')
        
        # Calculate p_t
        pt = torch.exp(-ce_loss)
        
        # Calculate alpha_t
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = self.alpha
            else:
                alpha_t = self.alpha[target]
        else:
            alpha_t = 1.0
        
        # Calculate focal loss
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss 