import torch


def _sigmoid(x):
    """Clamped sigmoid function

    Parameters
    ----------
    x : torch.tensor
        Tensor which to compute clamped sigmoid on

    Returns
    -------
    torch.tensor
        Return clamped sigmoid of input x
    """

    y = torch.clamp(x.sigmoid_(), min=1e-4, max=1 - 1e-4)
    return y


def _gather_feat(feat, ind, mask=None):
    """Collecting the features at given indexes

    Parameters
    ----------
    feat : torch.tensor
        Feature space in which to select features
        Shape: batch, HxW, num_classes
    ind : torch.tensor
        Indexes at which to collect features from feat
        Shape: num_classes, num_det=50
    mask : _type_, optional
        _description_, by default None

    Returns
    -------
    torch.tensor
        Returns selected features
        Shape: batch, num_det=50, num_classes
    """

    dim = feat.size(2)
    # make shape of ind to match shape of feat
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    # gathering features at ind along dimension 1
    feat = feat.gather(1, ind)
    # -> fetures are row wise feature vectors of the wh-matrix; 1 row is corresponding to one detection position in the image; 50 rows for 50 possible detections
    # mask is None!
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _tranpose_and_gather_feat(feat, ind):
    """Collecting features over different batches and classes, via the _gather_feat function after changing the shape accordingly.

    Parameters
    ----------
    feat : torch.tensor
        Some feature map (e.g. heatmap, wh, etc.).
        Shape: batch, num_classes, height and width of feature space
    ind : _type_
        The indexes per batch, where features are to be extracted
        Shape: batch, num_det=50
    Returns
    -------
    torch.tensor
        Returns selected features from feature space
        Shape: classes, num_det=50
    """

    # changing shape: moving classes/channel to the back: [batch, H, W, C]
    feat = feat.permute(0, 2, 3, 1).contiguous()
    # creating view in which height and width are in one dimension: [batch, HxW, C]
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat
