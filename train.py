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
from GLOBALS import *

def train(model, epoch, dataloader, optimizer):
    model.train()
    model.to(device)
    
    if epoch > 2000:
        for g in optimizer.param_groups:
            g['lr'] = 1e-5
    
    running_loss = 0.0
    count = 0
    for (in_img, gt_img, train_ids, ratios) in dataloader:
        # Twice as big because model upsamples.
        gt_img = gt_img.view((BATCH_SIZE, 3, ps*2, ps*2)).to(device).cpu().float()
        in_img = in_img.view((BATCH_SIZE, 4, ps, ps)).to(device).float()
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Get model outputs
        out_img = model(in_img)
        
        # Calculate loss
        loss = reduce_mean(out_img, gt_img)
        running_loss += loss.item()
        
        # Compute gradients and take step
        loss.backward()
        optimizer.step()
        
        if count % save_freq == 0 and count == 0:
            save_current_model(model, epoch, out_img[0], gt_img[0], train_ids[0].item(), ratios[0].item())
            count += 1

    return running_loss

