from __future__ import absolute_import

import torch
import torch.nn as nn

from .utils import _gather_feat, _tranpose_and_gather_feat


def _topk(heatmap, K=40, num_classes=1):
    """Find the top_k features in the heatmaps of all classes and return them

    Parameters
    ----------
    heatmap : torch.tensor
        Array of possible detections (one layer per object class)
    K : int, optional
        Number of features/detections to return, by default 40
    num_classes : int, optional
        Number of classes inside the heatmap, by default 1

    Returns
    -------
    (torch.tensor, torch.tensor, torch.tensor, torch.tensor, torch.tensor, torch.tensor)
        scores, indices, class identity, y-coordinate in feature space, x-coordinate in feature space, mask of k-features per class
    """

    N, C, H, W = heatmap.size()

    # 2d feature map -> 1d feature map
    topk_scores, topk_inds = torch.topk(heatmap.view(N, C, -1), K)
    topk_inds = topk_inds % (H * W)
    topk_ys = (topk_inds / W).int().float()
    topk_xs = (topk_inds % W).int().float()

    topk_score, topk_ind = torch.topk(topk_scores.view(N, -1), K)
    topk_clses = (topk_ind / K).int()

    topk_inds = _gather_feat(topk_inds.view(N, -1, 1), topk_ind).view(N, K)
    topk_ys = _gather_feat(topk_ys.view(N, -1, 1), topk_ind).view(N, K)
    topk_xs = _gather_feat(topk_xs.view(N, -1, 1), topk_ind).view(N, K)
    # selects 50 features from the 250 (50 selected per class)

    cls_inds_masks = torch.full((num_classes, K), False, dtype=torch.bool).to(topk_inds.device)

    for cls_id in range(num_classes):
        inds_masks = topk_clses==cls_id
        cls_inds_masks[cls_id] = inds_masks

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs, cls_inds_masks

def _max_pool(heat, kernel=3):
    """
    NCHW
    do max pooling operation
    """
    pad = (kernel - 1) // 2

    h_max = nn.functional.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)

    keep = (h_max == heat).float()

    return heat * keep

# FIXME get rid of num_classes (== C.size())
def mot_decode(heatmap,
               wh,
               reg=None,
               num_classes=1,
               cat_spec_wh=False,
               K=100):
    """Generates detections containing BB's, scores and classes for the peaks in the respective heatmaps

    Parameters
    ----------
    heatmap : torch.tensor
        heatmap of detection centers per class
        shape: batch, num_classes, feat_height, feat_width
    wh : torch.tensor
        width-height feature map, containing width and height of BB's at each possible center point in the feature space
        shape: batch, 2 or 2 per num_classes (cat_spec_wh), feat_height, feat_width
    reg : torch.tensor, optional
        feature map holding parameters for resizing BB to original image size, by default None
    num_classes : int, optional
        number of classes that the model trains on equal to C.size(), by default 1
    cat_spec_wh : bool, optional
        wether to predicting BB width and height separately per class per possible center point, by default False
    K : int, optional
        number of detections to predict per class, by default 50

    Returns
    -------
    (torch.tensor, torch.tensor, torch.tensor)
        returns a tuple of the detections containing (BB's, there scores and class affiliation), the indices of the detections, a mask of which indice in the feature space detections belongs to which class
    """

    N, C, H, W = heatmap.size()

    heatmap = _max_pool(heatmap)

    scores, inds, classes, ys, xs, cls_inds_masks = _topk(heatmap=heatmap, K=K, num_classes=num_classes)
    if reg is not None:
        reg = _tranpose_and_gather_feat(reg, inds)

        reg = reg.view(N, K, 2)
        xs = xs.view(N, K, 1) + reg[:, :, 0:1]
        ys = ys.view(N, K, 1) + reg[:, :, 1:2]
    else:
        xs = xs.view(N, K, 1) + 0.5
        ys = ys.view(N, K, 1) + 0.5
    wh = _tranpose_and_gather_feat(wh, inds)

    if cat_spec_wh:
        wh = wh.view(N, K, C, 2)
        clses_ind = classes.view(N, K, 1, 1).expand(N, K, 1, 2).long()
        wh = wh.gather(2, clses_ind).view(N, K, 2)
    else:
        wh = wh.view(N, K, 2)

    classes = classes.view(N, K, 1).float()
    scores = scores.view(N, K, 1)

    bboxes = torch.cat([xs - wh[..., 0:1] * 0.5,   # left    x1
                        ys - wh[..., 1:2] * 0.5,   # top     y1
                        xs + wh[..., 0:1] * 0.5,   # right   x2
                        ys + wh[..., 1:2] * 0.5],  # down    y2
                       dim=2)

    detections = torch.cat([bboxes, scores, classes], dim=2)

    return detections, inds, cls_inds_masks



def map2orig(dets, h_out, w_out, h_orig, w_orig, num_classes):
    """
    :param dets:
    :param h_out:
    :param w_out:
    :param h_orig:
    :param w_orig:
    :param num_classes:
    :return: dict of detections(key: cls_id)
    """

    def get_padding():
        """
        :return: pad_1, pad_2, pad_type('pad_x' or 'pad_y'), new_shape(w, h)
        """
        ratio_x = float(w_out) / w_orig
        ratio_y = float(h_out) / h_orig
        ratio = min(ratio_x, ratio_y)
        new_shape = (round(w_orig * ratio), round(h_orig * ratio))  # new_w, new_h
        
        pad_x = (w_out - new_shape[0]) * 0.5  # width padding
        pad_y = (h_out - new_shape[1]) * 0.5  # height padding
        top, bottom = round(pad_y - 0.1), round(pad_y + 0.1)
        left, right = round(pad_x - 0.1), round(pad_x + 0.1)
        if ratio == ratio_x:  # pad_y
            return top, bottom, 'pad_y', new_shape
        else:  # pad_x
            return left, right, 'pad_x', new_shape

    pad_1, pad_2, pad_type, new_shape = get_padding()

    dets = dets.detach().cpu().numpy()
    dets = dets.reshape(1, -1, dets.shape[2])  # default: 1×128×6
    dets = dets[0]  # 128×6

    dets_dict = {}

    if pad_type == 'pad_x':
        dets[:, 0] = (dets[:, 0] - pad_1) / new_shape[0] * w_orig  # x1
        dets[:, 2] = (dets[:, 2] - pad_1) / new_shape[0] * w_orig  # x2
        dets[:, 1] = dets[:, 1] / h_out * h_orig  # y1
        dets[:, 3] = dets[:, 3] / h_out * h_orig  # y2
    else:  # 'pad_y'
        dets[:, 0] = dets[:, 0] / w_out * w_orig  # x1
        dets[:, 2] = dets[:, 2] / w_out * w_orig  # x2
        dets[:, 1] = (dets[:, 1] - pad_1) / new_shape[1] * h_orig  # y1
        dets[:, 3] = (dets[:, 3] - pad_1) / new_shape[1] * h_orig  # y2

    classes = dets[:, -1]
    for cls_id in range(num_classes):
        inds = (classes == cls_id)
        dets_dict[cls_id] = dets[inds, :]

    return dets_dict