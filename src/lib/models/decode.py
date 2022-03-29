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
    # batch, cat, height, width = scores.size()
    N, C, H, W = heatmap.size()
      
    # topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)
    # 2d feature map -> 1d feature map
    topk_scores, topk_inds = torch.topk(heatmap.view(N, C, -1), K)

    # topk_inds = topk_inds % (height * width)
    topk_inds = topk_inds % (H * W)
    # topk_ys   = (topk_inds / width).int().float()
    # topk_xs   = (topk_inds % width).int().float()
    topk_ys = (topk_inds / W).int().float()
    topk_xs = (topk_inds % W).int().float()
      
    # topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_score, topk_ind = torch.topk(topk_scores.view(N, -1), K)
    topk_clses = (topk_ind / K).int()
    # topk_inds = _gather_feat(
    #     topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_inds = _gather_feat(topk_inds.view(N, -1, 1), topk_ind).view(N, K)
    # topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    # topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(N, -1, 1), topk_ind).view(N, K)
    topk_xs = _gather_feat(topk_xs.view(N, -1, 1), topk_ind).view(N, K)

    cls_inds_masks = torch.full((num_classes, K), False, dtype=torch.bool).to(topk_inds.device)
    for cls_id in range(num_classes):
        inds_masks = topk_clses==cls_id
        # cls_topk_inds = topk_inds[inds_masks]
        cls_inds_masks[cls_id] = inds_masks


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
# def mot_decode(heat, wh, reg=None, ltrb=False, K=100):
def mot_decode(heatmap,
               wh,
               reg=None,
               num_classes=2,
               cat_spec_wh=False,
               K=100):
    # batch, cat, height, width = heat.size()
    N, C, H, W = heatmap.size()

    # heat = torch.sigmoid(heat)
    # perform nms on heatmaps
    # heat = _nms(heat)
    heatmap = _max_pool(heatmap)

    # scores, inds, clses, ys, xs = _topk(heat, K=K)
    scores, inds, classes, ys, xs, cls_inds_masks = _topk(heatmap=heatmap, K=K, num_classes=num_classes)
    if reg is not None:
        reg = _tranpose_and_gather_feat(reg, inds)
        # reg = reg.view(batch, K, 2)
        # xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
        # ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
        reg = reg.view(N, K, 2)
        xs = xs.view(N, K, 1) + reg[:, :, 0:1]
        ys = ys.view(N, K, 1) + reg[:, :, 1:2]
    else:
        # xs = xs.view(batch, K, 1) + 0.5
        # ys = ys.view(batch, K, 1) + 0.5
        xs = xs.view(N, K, 1) + 0.5
        ys = ys.view(N, K, 1) + 0.5
    wh = _tranpose_and_gather_feat(wh, inds)
    # if ltrb:
    #     wh = wh.view(batch, K, 4)
    # else:
    #     wh = wh.view(batch, K, 2)
    if cat_spec_wh:
        wh = wh.view(N, K, C, 2)
        clses_ind = classes.view(N, K, 1, 1).expand(N, K, 1, 2).long()
        wh = wh.gather(2, clses_ind).view(N, K, 2)
    else:
        wh = wh.view(N, K, 2)
    # clses = clses.view(batch, K, 1).float()
    # scores = scores.view(batch, K, 1)
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
