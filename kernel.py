import numpy as np
import torch
import torch.nn as nn


def adaptive_isotropic_gaussian_kernel(xs, ys, h_min=1e-3):
    
    Kx, D = list(xs.size())[-2:]
    Ky, D2 = list(ys.size())[-2:]
    assert D == D2

    leading_shape = list(xs.size()[:-2])

    diff = xs[:,:,None,:] - ys[:,None,:,:]
    dist_sq = torch.sum(diff**2, dim=-1, keepdims=False)

    input_shape = (leading_shape[0],Kx*Ky)

    values,_ = torch.topk(
        input = dist_sq.reshape(
            input_shape
        ),
        k=(Kx * Ky // 2 + 1),
        largest = True,
        sorted=True
    )

    medians_sq = values[..., -1]  # ... (shape) (last element is the median)

    h = medians_sq / np.log(Kx)  # ... (shape)
    #print(h.shape)
    #print(torch.Tensor([h_min]).shape)
    #h = torch.max(torch.Tensor([h,torch.Tensor([h_min])]))
    h = torch.max(h,torch.Tensor([h_min]))
    h = h.detach()  # Just in case.
    h_expanded_twice = (h.unsqueeze(-1)).unsqueeze(-1)

    kappa = torch.exp(-dist_sq / h_expanded_twice)  # ... x Kx x Ky

    h_expanded_thrice = h_expanded_twice.unsqueeze(-1)
    
    kappa_expanded = kappa.unsqueeze(-1)

    kappa_grad = -2 * diff / h_expanded_thrice * kappa_expanded

    return {"output": kappa, "gradient": kappa_grad}