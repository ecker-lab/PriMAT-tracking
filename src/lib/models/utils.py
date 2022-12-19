from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

def _sigmoid(x):
  y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
  return y

def _gather_feat(feat, ind, mask=None):
    #        [1, 41344, 4], [4,50]
    dim  = feat.size(2)# 4
    #           [4, 50, ' ']       [4,           50,         '4']
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    #[1, 50, 4] = [1, 41344, 4]      [4, 50, 4]
    feat = feat.gather(1, ind)# 1 <- axis along wich to index, ind <- indicies of elements to gather
    # -> fetures are row wise feature vectors of the wh-matrix; 1 row is corresponding to one detection position in the image; 50 rows for 50 possible detections
    #mask is None!
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat

def _tranpose_and_gather_feat(feat, ind):
    # feat (batch x c x h x w)
    # ind (batch, det)
    feat = feat.permute(0, 2, 3, 1).contiguous()#[batch, h, w, c]
    feat = feat.view(feat.size(0), -1, feat.size(3))#[batch, HxW, c]
    feat = _gather_feat(feat, ind)#[1, 41344, 4], [4, 50]
    return feat

def flip_tensor(x):
    return torch.flip(x, [3])
    # tmp = x.detach().cpu().numpy()[..., ::-1].copy()
    # return torch.from_numpy(tmp).to(x.device)

def flip_lr(x, flip_idx):
  tmp = x.detach().cpu().numpy()[..., ::-1].copy()
  shape = tmp.shape
  for e in flip_idx:
    tmp[:, e[0], ...], tmp[:, e[1], ...] = \
      tmp[:, e[1], ...].copy(), tmp[:, e[0], ...].copy()
  return torch.from_numpy(tmp.reshape(shape)).to(x.device)

def flip_lr_off(x, flip_idx):
  tmp = x.detach().cpu().numpy()[..., ::-1].copy()
  shape = tmp.shape
  tmp = tmp.reshape(tmp.shape[0], 17, 2, 
                    tmp.shape[2], tmp.shape[3])
  tmp[:, :, 0, :, :] *= -1
  for e in flip_idx:
    tmp[:, e[0], ...], tmp[:, e[1], ...] = \
      tmp[:, e[1], ...].copy(), tmp[:, e[0], ...].copy()
  return torch.from_numpy(tmp.reshape(shape)).to(x.device)