import torch
import torch.nn as nn
import torch.nn.functional as F

def pool_feat(f1, f2, pool_dim, normalize_feat=False):
    assert f1.shape == f2.shape, (f1.shape, f2.shape) # (N, C, H, W)

    # If already pooled, return the original features
    if f1.ndim == 2:
        return f1, f2
    
    # Do the pooling and move the channel dim to the last
    if pool_dim == 'hw':
        f1 = f1.mean(dim=(2, 3)) # (N, C)
        f2 = f2.mean(dim=(2, 3)) # (N, C)
    elif pool_dim == 'c':
        f1 = f1.mean(dim=1).reshape(f1.shape[0], -1) # (N, H*W)
        f2 = f2.mean(dim=1).reshape(f2.shape[0], -1) # (N, H*W)
    elif pool_dim == 'h':
        f1 = f1.mean(dim=2).reshape(f1.shape[0], -1) # (N, C*W)
        f2 = f2.mean(dim=2).reshape(f2.shape[0], -1) # (N, C*W)
    elif pool_dim == 'w':
        f1 = f1.mean(dim=3).reshape(f1.shape[0], -1) # (N, C*H)
        f2 = f2.mean(dim=3).reshape(f2.shape[0], -1) # (N, C*H)
    elif pool_dim == 'flatten':
        f1 = f1.transpose(1, 3) # (N, W, H, C)
        f2 = f2.transpose(1, 3) # (N, W, H, C)
    else:
        raise ValueError("Unknown pooling dimension: {}".format(pool_dim))
        
    # Treat each example as an unit
    f1 = f1.reshape(f1.shape[0], -1) # (N, -1)
    f2 = f2.reshape(f2.shape[0], -1) # (N, -1)
    
    # Normalize features if needed
    if normalize_feat:
        f1 = F.normalize(f1, p=2, dim=1)
        f2 = F.normalize(f2, p=2, dim=1)

    return (f1, f2)

def match_loss(fs, loss_type):
    f1, f2 = fs
    # Compute the loss according to the loss type
    if loss_type == 'mse':
        loss = F.mse_loss(f1, f2)
    elif loss_type == 'rmse':
        loss = F.mse_loss(f1, f2) ** 0.5
    elif loss_type == 'abs':
        loss = F.l1_loss(f1, f2)
    elif loss_type == 'mfro':
        # Mean of Frobenius norm, normalized by the number of elements
        # loss = torch.mean(torch.frobenius_norm(f1 - f2, dim=-1)) / (float(f1.shape[-1]) ** 0.5)
        loss = torch.mean(torch.frobenius_norm(f1 - f2, dim=1)) / (float(f1.shape[1]) ** 0.5)
    elif loss_type == "cos":
        loss = 1 - F.cosine_similarity(f1, f2, dim=1).mean()
    elif loss_type == 'kl':
        loss = F.kl_div(f1, f2)
    else:
        raise ValueError("Unknown loss type: {}".format(loss_type))

    return loss
