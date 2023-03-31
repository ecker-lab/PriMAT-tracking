import numpy as np
import torch
import torch.nn as nn
from models.utils import _tranpose_and_gather_feat


class General_Classification(nn.Module):
    # opts.K needed????
    def __init__(self, num_cls, clsID4GC=0, emb_dim=128):
        super().__init__()
        self.emb_dim =  emb_dim
        self.num_cls = num_cls
        self.clsID4GC = clsID4GC
        
        self.gc_classifier = nn.Linear(self.emb_dim, self.num_cls, bias=True)
        self.sm = nn.Softmax(dim=1)
        

    def forward(self, gc_features, cls_id_map, target=None):
        """ additional function to forward if gc head is active to give feature vector for monkey detection out of feature map
        
        Parameters
        ----------
        gc_features : torch.tensor
        # TODO fix
            gc head feature map
        cls_id_map : torch.tensor
            map which entrys of feature map correspond to each specific class ID
        target : torch.tensor / None
            ground truth of classification task / None for inference

        Returns
        -------
        torch.tensor
            prediction of a classification task
        """
        if target is None:
            inds = cls_id_map
            feat = _tranpose_and_gather_feat(gc_features, inds)
            feat = feat.squeeze(0)
        else:
            inds = torch.where(cls_id_map == self.clsID4GC)
            if inds[0].shape[0] == 0:
                print(target.numel())
                #stand_in = torch.ones((target.numel(),18), dtype=torch.float32).to(target)
                #stand_in = torch.tensor([0, 0, 0, 0, 1]).expand(target.numel(), 5).to(target)
                #return stand_in.type(torch.float32)
                return torch.Tensor()
                # return torch.zeros_like(target)
            feat = gc_features[inds[0],:,inds[2],inds[3]]
            # feat = self.emb_scale * F.normalize(feat)

        pred = self.gc_classifier(feat).contiguous()
        
        if target is None:
            return self.sm(pred).detach()
        
        # catch missing predictions for case that monkey is not in frame
        pred_fix = torch.zeros((target.size()[0], self.num_cls), dtype=torch.float32).to(pred.device)
        # not in Classes filler label
        # fill = torch.zeros((1,self.num_cls)).to(pred.device)
        # fill[0,-1] = 1
        #pred_fix += fill
        pred_fix[inds[0]] = pred
        return pred_fix