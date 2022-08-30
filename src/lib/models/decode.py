from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from .utils import _gather_feat, _tranpose_and_gather_feat

def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


def _topk_channel(scores, K=40):
      batch, cat, height, width = scores.size()
      
      topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

      topk_inds = topk_inds % (height * width)
      topk_ys   = (topk_inds / width).int().float()
      topk_xs   = (topk_inds % width).int().float()

      return topk_scores, topk_inds, topk_ys, topk_xs


# altered by mcmot
def _topk(heatmap, K=40, num_classes=1):
    N, C, H, W = heatmap.size()
    
    # 2d feature map -> 1d feature map
    topk_scores, topk_inds = torch.topk(heatmap.view(N, C, -1), K)# 4, 5, HxW
    #[4, 5, 50] <- value, [4, 5, 50] <- which pixel
    topk_inds = topk_inds % (H * W)
    topk_ys = (topk_inds / W).int().float()
    topk_xs = (topk_inds % W).int().float()
      
    topk_score, topk_ind = torch.topk(topk_scores.view(N, -1), K)
    # [4, 50] <- value, [4, 50] <- which class
    topk_clses = (topk_ind / K).int()

    # ([4, 250, 1], [4, 50]).view(4, 50) 
    topk_inds = _gather_feat(topk_inds.view(N, -1, 1), topk_ind).view(N, K)# <- returns (4,50,1)
    topk_ys = _gather_feat(topk_ys.view(N, -1, 1), topk_ind).view(N, K)
    topk_xs = _gather_feat(topk_xs.view(N, -1, 1), topk_ind).view(N, K)
    # selects 50 features from the 250 (50 selected per class)

    # added N to tensor size
    cls_inds_masks = torch.full((num_classes, K), False, dtype=torch.bool).to(topk_inds.device)
    #[5, 4, 50]
    #num_classes: 5
    for cls_id in range(num_classes):
        #cls_id: 0
        #topk_clses: [4, 50] <- batch_size, K
        inds_masks = topk_clses==cls_id
        #[4,50]
        # cls_topk_inds = topk_inds[inds_masks]
        # 
        cls_inds_masks[cls_id] = inds_masks
        #([5, 4, 50])[0] -> [4, 50] == [4, 50]

    #                  [4,50]                                   [5,4,50]
    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs, cls_inds_masks


# new from mcmot
def _max_pool(heat, kernel=3):
    """
    NCHW
    do max pooling operation
    """
    # print("heat.shape: ", heat.shape)  # default: torch.Size([1, 1, 152, 272])
    pad = (kernel - 1) // 2

    h_max = nn.functional.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    # print("h_max.shape: ", h_max.shape)  # default: torch.Size([1, 1, 152, 272])

    keep = (h_max == heat).float()  # 将boolean类型的Tensor转换成Float类型的Tensor
    # print("keep.shape: ", keep.shape, "keep:\n", keep)
    return heat * keep


# altered by mcmot
def mot_decode(heatmap,
               wh,
               reg=None,
               num_classes=2,
               cat_spec_wh=False,
               K=100):

    N, C, H, W = heatmap.size()

    # heat = torch.sigmoid(heat)
    # perform nms on heatmaps
    # heat = _nms(heat)
    heatmap = _max_pool(heatmap)

    scores, inds, classes, ys, xs, cls_inds_masks = _topk(heatmap=heatmap, K=K, num_classes=num_classes)
    if reg is not None:
        reg = _tranpose_and_gather_feat(reg, inds)

        reg = reg.view(N, K, 2)
        xs = xs.view(N, K, 1) + reg[:, :, 0:1]
        ys = ys.view(N, K, 1) + reg[:, :, 1:2]
    else:
        # xs = xs.view(batch, K, 1) + 0.5
        # ys = ys.view(batch, K, 1) + 0.5
        xs = xs.view(N, K, 1) + 0.5
        ys = ys.view(N, K, 1) + 0.5
    wh = _tranpose_and_gather_feat(wh, inds)

    if cat_spec_wh:
        wh = wh.view(N, K, C, 2)
        clses_ind = classes.view(N, K, 1, 1).expand(N, K, 1, 2).long()
        wh = wh.gather(2, clses_ind).view(N, K, 2)
    else:
        wh = wh.view(N, K, 2)

    classes = classes.view(N, K, 1).float()  # 目标类别
    scores = scores.view(N, K, 1)
    # if ltrb:
    #     bboxes = torch.cat([xs - wh[..., 0:1],
    #                         ys - wh[..., 1:2],
    #                         xs + wh[..., 2:3],
    #                         ys + wh[..., 3:4]], dim=2)
    # else:
    #     bboxes = torch.cat([xs - wh[..., 0:1] / 2,
    #                         ys - wh[..., 1:2] / 2,
    #                         xs + wh[..., 0:1] / 2,
    #                         ys + wh[..., 1:2] / 2], dim=2)
    bboxes = torch.cat([xs - wh[..., 0:1] * 0.5,   # left    x1
                        ys - wh[..., 1:2] * 0.5,   # top     y1
                        xs + wh[..., 0:1] * 0.5,   # right   x2
                        ys + wh[..., 1:2] * 0.5],  # down    y2
                       dim=2)
    # detections = torch.cat([bboxes, scores, clses], dim=2)
    detections = torch.cat([bboxes, scores, classes], dim=2)

    # return detections, inds
    return detections, inds, cls_inds_masks
