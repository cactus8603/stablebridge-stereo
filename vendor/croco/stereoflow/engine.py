# Copyright (C) 2022-present Naver Corporation. All rights reserved.

# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
# Adapted to retain inference helpers only.

import numpy as np
import torch
import torch.nn.functional as F

def split_prediction_conf(predictions, with_conf=False):
    if not with_conf:
        return predictions, None
    conf = predictions[:,-1:,:,:]
    predictions = predictions[:,:-1,:,:]
    return predictions, conf


def _resize_img(img, new_size):
    return F.interpolate(img, size=new_size, mode='bicubic', align_corners=False)


def _resize_stereo_or_flow(data, new_size):
    assert data.ndim==4
    assert data.size(1) in [1,2]
    scale_x = new_size[1]/float(data.size(3))
    out = F.interpolate(data, size=new_size, mode='bicubic', align_corners=False)
    out[:,0,:,:] *= scale_x
    if out.size(1)==2:
        scale_y = new_size[0]/float(data.size(2))        
        out[:,1,:,:] *= scale_y
        print(scale_x, new_size, data.shape)
    return out


@torch.no_grad()
def tiled_pred(model, criterion, img1, img2, gt,
               overlap=0.5, bad_crop_thr=0.05,
               downscale=False, crop=512, ret='loss',
               conf_mode='conf_expsigmoid_10_5', with_conf=False, 
               return_time=False):
                     
    # for each image, we are going to run inference on many overlapping patches
    # then, all predictions will be weighted-averaged
    if gt is not None:
        B, C, H, W = gt.shape
    else:
        B, _, H, W = img1.shape
        C = model.head.num_channels-int(with_conf)
    win_height, win_width = crop[0], crop[1]
    
    # upscale to be larger than the crop
    do_change_scale =  H<win_height or W<win_width
    if do_change_scale: 
        upscale_factor = max(win_width/W, win_height/W)
        original_size = (H,W)
        new_size = (round(H*upscale_factor),round(W*upscale_factor))
        img1 = _resize_img(img1, new_size)
        img2 = _resize_img(img2, new_size)
        # resize gt just for the computation of tiled losses
        if gt is not None: gt = _resize_stereo_or_flow(gt, new_size)
        H,W = img1.shape[2:4]
        
    if conf_mode.startswith('conf_expsigmoid_'): # conf_expsigmoid_30_10
        beta, betasigmoid = map(float, conf_mode[len('conf_expsigmoid_'):].split('_'))
    elif conf_mode.startswith('conf_expbeta'): # conf_expbeta3
        beta = float(conf_mode[len('conf_expbeta'):])
    else:
        raise NotImplementedError(f"conf_mode {conf_mode} is not implemented")

    def crop_generator():
        for sy in _overlapping(H, win_height, overlap):
          for sx in _overlapping(W, win_width, overlap):
            yield sy, sx, sy, sx, True

    # keep track of weighted sum of prediction*weights and weights
    accu_pred = img1.new_zeros((B, C, H, W)) # accumulate the weighted sum of predictions 
    accu_conf = img1.new_zeros((B, H, W)) + 1e-16 # accumulate the weights 
    accu_c = img1.new_zeros((B, H, W)) # accumulate the weighted sum of confidences ; not so useful except for computing some losses

    tiled_losses = []
    
    if return_time:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

    for sy1, sx1, sy2, sx2, aligned in crop_generator():
        # compute optical flow there
        pred =  model(_crop(img1,sy1,sx1), _crop(img2,sy2,sx2))
        pred, predconf = split_prediction_conf(pred, with_conf=with_conf)
        
        if gt is not None: gtcrop = _crop(gt,sy1,sx1)
        if criterion is not None and gt is not None: 
            tiled_losses.append( criterion(pred, gtcrop).item() if predconf is None else criterion(pred, gtcrop, predconf).item() )
        
        if conf_mode.startswith('conf_expsigmoid_'):
            conf = torch.exp(- beta * 2 * (torch.sigmoid(predconf / betasigmoid) - 0.5)).view(B,win_height,win_width)
        elif conf_mode.startswith('conf_expbeta'):
            conf = torch.exp(- beta * predconf).view(B,win_height,win_width)
        else:
            raise NotImplementedError
                        
        accu_pred[...,sy1,sx1] += pred * conf[:,None,:,:]
        accu_conf[...,sy1,sx1] += conf
        accu_c[...,sy1,sx1] += predconf.view(B,win_height,win_width) * conf 
        
    pred = accu_pred / accu_conf[:, None,:,:]
    c = accu_c / accu_conf
    assert not torch.any(torch.isnan(pred))

    if return_time:
        end.record()
        torch.cuda.synchronize()
        time = start.elapsed_time(end)/1000.0 # this was in milliseconds

    if do_change_scale:
        pred = _resize_stereo_or_flow(pred, original_size)
    
    if return_time:
        return pred, torch.mean(torch.tensor(tiled_losses)), c, time
    return pred, torch.mean(torch.tensor(tiled_losses)), c


def _overlapping(total, window, overlap=0.5):
    assert total >= window and 0 <= overlap < 1, (total, window, overlap)
    num_windows = 1 + int(np.ceil( (total - window) / ((1-overlap) * window) ))
    offsets = np.linspace(0, total-window, num_windows).round().astype(int)
    yield from (slice(x, x+window) for x in offsets)


def _crop(img, sy, sx):
    B, THREE, H, W = img.shape
    if 0 <= sy.start and sy.stop <= H and 0 <= sx.start and sx.stop <= W:
        return img[:,:,sy,sx]
    l, r = max(0,-sx.start), max(0,sx.stop-W)
    t, b = max(0,-sy.start), max(0,sy.stop-H)
    img = torch.nn.functional.pad(img, (l,r,t,b), mode='constant')
    return img[:, :, slice(sy.start+t,sy.stop+t), slice(sx.start+l,sx.stop+l)]
