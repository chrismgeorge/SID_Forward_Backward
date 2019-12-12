import time, pdb, os
import numpy as np
from GLOBALS import device, ps, BATCH_SIZE, save_freq
from utils import pack_raw, reduce_mean, save_current_model
import torch
import os, scipy.io, pdb

import glob
import rawpy
import torch
import numpy as np
# from canny import canny_f
from GLOBALS import *

# def train(model, epoch, dataloader, optimizer):
#     model.train()
#     model.to(device)
    
#     if epoch > 2000:
#         for g in optimizer.param_groups:
#             g['lr'] = 1e-5
    
#     running_loss = 0.0
#     count = 0
#     for (in_img, gt_img, train_ids, ratios) in dataloader:
#         # Twice as big because model upsamples.
#         gt_img = gt_img.view((BATCH_SIZE, 3, ps*2, ps*2)).to(device).cpu().float()
#         in_img = in_img.view((BATCH_SIZE, 4, ps, ps)).to(device).float()
        
#         # Zero gradients
#         optimizer.zero_grad()
        
#         # Get model outputs
#         out_img = model(in_img)
        
#         # Calculate loss
#         loss = reduce_mean(out_img, gt_img)
#         running_loss += loss.item()
        
#         # Compute gradients and take step
#         loss.backward()
#         optimizer.step()
        
#         if count % save_freq == 0 and count == 0:
#             save_current_model(model, epoch, out_img[0], gt_img[0], train_ids[0].item(), ratios[0].item())
#             count += 1

#     return running_loss

import numpy as np
import cv2
 
def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

def train_experimental(model, epoch, dataloader, optimizer):
    model.train()
    model.to(device)
    
    if epoch > 10:
        for g in optimizer.param_groups:
            g['lr'] = 1e-5
    if epoch > 2000:
        for g in optimizer.param_groups:
            g['lr'] = 1e-6
    
    running_loss = 0.0
    count = 0
    for (in_img, gt_img, train_ids, ratios) in dataloader:
        # Twice as big because model upsamples.
        gt_img = gt_img.view((BATCH_SIZE, 3, ps, ps)).to(device).float()
        in_img = in_img.view((BATCH_SIZE, 3, ps, ps)).to(device).float()
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Get model outputs
        out_img, out_edge = model(in_img)

        # Calculate loss
        new_gt = gt_img.view([256, 256, 3]).cpu().numpy()
        new_gt = np.uint8(rgb2gray(new_gt))
        gt_edge = auto_canny(new_gt)
        gt_edge = torch.tensor(gt_edge).to(device).float()
        out_edge = out_edge.view((ps, ps)).to(device)
        
        loss = reduce_mean(out_img, gt_img, gt_edge, out_edge)
        running_loss += loss.item()
        
        # Compute gradients and take step
        loss.backward()
        optimizer.step()
        
        if count % save_freq == 0 and count == 0:
            save_current_model(model, epoch, out_img[0], gt_img[0], train_ids[0].item(), ratios[0].item())
            count += 1

    return running_loss

