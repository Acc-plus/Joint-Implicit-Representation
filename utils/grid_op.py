import torch
import torch.nn as nn
import numpy as np
from devices import *

def grid_patch(center, radius, resolution):
    x, y = center
    xflat = torch.arange(x - radius, x + radius, 2 * radius / resolution, device=device)
    yflat = torch.arange(y - radius, y + radius, 2 * radius / resolution, device=device)
    xflat = xflat.expand(resolution, resolution)
    yflat = yflat.view(resolution, 1).expand(resolution, resolution)
    return torch.stack([xflat, yflat], dim=-1)



def find_global_maxima(patch, resolution):
    resolutiond = resolution - 1
    resolutiondd = resolutiond - 1
    imdif = torch.zeros(8, patch.shape[0], resolution, resolution, device=device)
    imdif[:] = patch
    imdif[0, :, 1:resolutiond, 1:resolutiond] -= patch[:, :resolutiondd, :resolutiondd]
    imdif[1, :, 1:resolutiond, 1:resolutiond] -= patch[:, 1:resolutiond, :resolutiondd]
    imdif[2, :, 1:resolutiond, 1:resolutiond] -= patch[:, 2:resolution, :resolutiondd]
    imdif[3, :, 1:resolutiond, 1:resolutiond] -= patch[:, 2:resolution, 1:resolutiond]
    imdif[4, :, 1:resolutiond, 1:resolutiond] -= patch[:, 2:resolution, 2:resolution]
    imdif[5, :, 1:resolutiond, 1:resolutiond] -= patch[:, 1:resolutiond, 2:resolution]
    imdif[6, :, 1:resolutiond, 1:resolutiond] -= patch[:, :resolutiondd, 2:resolution]
    imdif[7, :, 1:resolutiond, 1:resolutiond] -= patch[:, :resolutiondd, 1:resolutiond]
    imb = imdif >= 0.
    maxima = imb[0] & imb[1] & imb[2] & imb[3] & imb[4] & imb[5] & imb[6] & imb[7] & (patch > 0.5)
    return maxima


def find_local_maximum(n_imgs, n_corners, coords, patchvalue):
    patchvalue = patchvalue.view(n_imgs*n_corners, 32*32)
    sel = torch.argmax(patchvalue, dim = 1).unsqueeze(1)
    coords = coords.view(n_imgs*n_corners, 32*32, 2)
    cx = coords[:, :, 0].gather(1, sel)
    cy = coords[:, :, 1].gather(1, sel)
    local_max = torch.cat([cx, cy], dim=-1)

    radius = 0.2
    resolution = 64
    interval = radius * 2. / resolution
    flat = torch.arange(0, 2 * radius, interval, device=device)
    xflat = flat.expand(resolution, resolution)
    yflat = flat.view(resolution, 1).expand(resolution, resolution)
    adder = torch.stack([xflat, yflat], dim=-1).expand(n_imgs, n_corners, resolution, resolution, 2)
    # B, 8, R, R, 2
    left_corner = local_max.view(n_imgs, n_corners, 2) - radius
    left_corner = left_corner.unsqueeze(2).unsqueeze(2)
    # B, 8, 1, 1, 2
    patchs = left_corner + adder
    # B, 8, R, R, 2
    return patchs.view(n_imgs*n_corners, resolution, resolution, 2)
    # Bx8, R, R, 2
    # Input to Implicit Field For Discriminator
    

def shuffle_padding_truncation(n_imgs, n_corners, coords, selects):
    # reshape the patch from B, 8, H, W to Bx8, 1, H, W
    # concat two patch to generate 2 channel image Bx8, 2, H, W
    # pass the image to discriminator

    corner_select = torch.zeros((n_imgs, n_corners, 2), device=device)
    corner_mask = torch.ones((n_imgs, n_corners), dtype=torch.bool, device=device)
    # B, 8, 2
    for i in range(selects.shape[0]):
        n_sel = selects[i].sum()
        if (n_sel > n_corners):
            rperm = torch.randperm(n_sel, device=device)
            sel = coords[i][selects[i]][rperm]
            corner_select[i, :] = sel[:n_corners]
        else:
            sel = coords[i][selects[i]]
            corner_select[i, :n_sel] = sel
            corner_mask[i, n_sel:] = False
    # import pdb; pdb.set_trace()
    # np.save('cs.npy', corner_select.cpu().numpy())


    radius = 0.02
    resolution = 32
    interval = radius * 2. / resolution
    flat = torch.arange(0, 2 * radius, interval, device=device)
    xflat = flat.expand(resolution, resolution)
    yflat = flat.view(resolution, 1).expand(resolution, resolution)
    adder = torch.stack([xflat, yflat], dim=-1).expand(n_imgs, n_corners, resolution, resolution, 2)
    # B, 8, R, R, 2
    left_corner = corner_select - radius
    left_corner = left_corner.unsqueeze(2).unsqueeze(2)
    # B, 8, 1, 1, 2
    patchs = left_corner + adder
    # B, 8, R, R, 2
    return patchs.view(n_imgs*n_corners, resolution, resolution, 2), corner_mask
    # Bx8, R, R, 2
    # Input to Implicit Field
    
