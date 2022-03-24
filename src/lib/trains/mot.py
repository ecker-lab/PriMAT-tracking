from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from fvcore.nn import sigmoid_focal_loss_jit

from models.losses import FocalLoss, TripletLoss
from models.losses import RegL1Loss, RegLoss, NormRegL1Loss, RegWeightedL1Loss
from models.decode import mot_decode
from models.utils import _sigmoid, _tranpose_and_gather_feat
from utils.post_process import ctdet_post_process
from .base_trainer import BaseTrainer


class MotLoss(torch.nn.Module):
    def __init__(self, opt):
        super(MotLoss, self).__init__()
        self.crit = torch.nn.MSELoss() if opt.mse_loss else FocalLoss()
        self.crit_reg = RegL1Loss() if opt.reg_loss == 'l1' else \
            RegLoss() if opt.reg_loss == 'sl1' else None
        self.crit_wh = torch.nn.L1Loss(reduction='sum') if opt.dense_wh else \
            NormRegL1Loss() if opt.norm_wh else \
                RegWeightedL1Loss() if opt.cat_spec_wh else self.crit_reg
        self.opt = opt
        self.emb_dim = opt.reid_dim
        self.nID = opt.nID
        self.classifier = nn.Linear(self.emb_dim, self.nID)
        if opt.id_loss == 'focal':
            torch.nn.init.normal_(self.classifier.weight, std=0.01)
            prior_prob = 0.01
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            torch.nn.init.constant_(self.classifier.bias, bias_value)
        self.IDLoss = nn.CrossEntropyLoss(ignore_index=-1)
        self.emb_scale = math.sqrt(2) * math.log(self.nID - 1)
        self.s_det = nn.Parameter(-1.85 * torch.ones(1))
        self.s_id = nn.Parameter(-1.05 * torch.ones(1))

    def forward(self, outputs, batch):
        opt = self.opt
        hm_loss, wh_loss, off_loss, id_loss = 0, 0, 0, 0
        for s in range(opt.num_stacks):
            output = outputs[s]
            if not opt.mse_loss:
                output['hm'] = _sigmoid(output['hm'])

            hm_loss += self.crit(output['hm'], batch['hm']) / opt.num_stacks
            if opt.wh_weight > 0:
                # FIXME this is from MCMOT. should we include it?
                # if opt.dense_wh:
                #     mask_weight = batch['dense_wh_mask'].sum() + 1e-4
                #     wh_loss += (self.crit_wh(output['wh'] * batch['dense_wh_mask'],
                #                              batch['dense_wh'] * batch['dense_wh_mask']) /
                #                 mask_weight) / opt.num_stacks
                # else:  #
                wh_loss += self.crit_reg(
                    output['wh'], batch['reg_mask'],
                    batch['ind'], batch['wh']) / opt.num_stacks

            if opt.reg_offset and opt.off_weight > 0:
                off_loss += self.crit_reg(output['reg'], batch['reg_mask'],
                                          batch['ind'], batch['reg']) / opt.num_stacks

            if opt.id_weight > 0:
                id_head = _tranpose_and_gather_feat(output['id'], batch['ind'])
                id_head = id_head[batch['reg_mask'] > 0].contiguous()
                id_head = self.emb_scale * F.normalize(id_head)
                id_target = batch['ids'][batch['reg_mask'] > 0]

                cls_id_pred = self.classifier(id_head).contiguous()
                if self.opt.id_loss == 'focal':
                    id_target_one_hot = cls_id_pred.new_zeros((id_head.size(0), self.nID)).scatter_(1,
                                                                                                  id_target.long().view(
                                                                                                      -1, 1), 1)
                    id_loss += sigmoid_focal_loss_jit(cls_id_pred, id_target_one_hot,
                                                      alpha=0.25, gamma=2.0, reduction="sum"
                                                      ) / cls_id_pred.size(0)
                else:
                    id_loss += self.IDLoss(cls_id_pred, id_target)

        det_loss = opt.hm_weight * hm_loss + opt.wh_weight * wh_loss + opt.off_weight * off_loss
        if opt.multi_loss == 'uncertainty':
            loss = torch.exp(-self.s_det) * det_loss + torch.exp(-self.s_id) * id_loss + (self.s_det + self.s_id)
            loss *= 0.5
        else:
            loss = det_loss + 0.1 * id_loss

        loss_stats = {'loss': loss, 'hm_loss': hm_loss,
                      'wh_loss': wh_loss, 'off_loss': off_loss, 'id_loss': id_loss}
        return loss, loss_stats


class McMotLoss(torch.nn.Module):
    def __init__(self, opt):
        super(McMotLoss, self).__init__()

        self.crit = torch.nn.MSELoss() if opt.mse_loss else FocalLoss()
        self.crit_reg = RegL1Loss() if opt.reg_loss == 'l1' else \
            RegLoss() if opt.reg_loss == 'sl1' else None
        self.crit_wh = torch.nn.L1Loss(reduction='sum') if opt.dense_wh else \
            NormRegL1Loss() if opt.norm_wh else \
                RegWeightedL1Loss() if opt.cat_spec_wh else self.crit_reg
        self.opt = opt

        self.emb_dim = opt.reid_dim

        self.nID = opt.nID

        self.classifiers = nn.ModuleDict()
        for cls_id, nID in self.nID_dict.items():
            self.classifiers[str(cls_id)] = nn.Linear(self.emb_dim, nID)
        
        # self.IDLoss = nn.CrossEntropyLoss(ignore_index=-1)
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-1)
        if opt.id_loss == 'focal':
            for cls_id, nID in self.nID_dict.items():
                torch.nn.init.normal_(self.classifiers[str(cls_id)].weight, std=0.01)
                prior_prob = 0.01
                bias_value = -math.log((1 - prior_prob) / prior_prob)
                torch.nn.init.constant_(self.classifiers[str(cls_id)].bias, bias_value)
        # self.emb_scale = math.sqrt(2) * math.log(self.nID - 1)
        self.emb_scale_dict = dict()
        for cls_id, nID in self.nID_dict.items():
            self.emb_scale_dict[cls_id] = math.sqrt(2) * math.log(nID - 1)
        # track reid
        self.s_id = nn.Parameter(-1.05 * torch.ones(1))
        # scale factor of detection loss
        self.s_det = nn.Parameter(-1.85 * torch.ones(1))


    def forward(self, outputs, batch):
        opt = self.opt
        hm_loss, wh_loss, off_loss, reid_loss = 0, 0, 0, 0
        for s in range(opt.num_stacks):
            output = outputs[s]
            if not opt.mse_loss:
                output['hm'] = _sigmoid(output['hm'])

            hm_loss += self.crit(output['hm'], batch['hm']) / opt.num_stacks
            if opt.wh_weight > 0:
                # FIXME this is from MCMOT. should we include it?
                # if opt.dense_wh:
                #     mask_weight = batch['dense_wh_mask'].sum() + 1e-4
                #     wh_loss += (self.crit_wh(output['wh'] * batch['dense_wh_mask'],
                #                              batch['dense_wh'] * batch['dense_wh_mask']) /
                #                 mask_weight) / opt.num_stacks
                # else:  #
                wh_loss += self.crit_reg(
                    output['wh'], batch['reg_mask'],
                    batch['ind'], batch['wh']) / opt.num_stacks

            if opt.reg_offset and opt.off_weight > 0:
                off_loss += self.crit_reg(output['reg'], batch['reg_mask'],
                                          batch['ind'], batch['reg']) / opt.num_stacks
            # if is irrelevant
            if opt.id_weight > 0:
                cls_id_map = batch['cls_id_map']
                for cls_id, id_num in self.nID_dict.items():
                    inds = torch.where(cls_id_map == cls_id)
                    # skip not relevant classes
                    if inds[0].shape[0] == 0:
                        continue
                    cls_id_head = output['id'][inds[0], :, inds[2], inds[3]]
                    # cls_id_head = _tranpose_and_gather_feat(output['id'], batch['ind'])
                    # cls_id_head = id_head[batch['reg_mask'] > 0].contiguous()
                    cls_id_head = self.emb_scale_dict[cls_id] * F.normalize(cls_id_head)
                    # cls_id_target = batch['ids'][batch['reg_mask'] > 0]
                    cls_id_target = batch['cls_tr_ids'][inds[0], cls_id, inds[2], inds[3]]

                    cls_id_pred = self.classifiers[str(cls_id)].forward(cls_id_head).contiguous()

                    reid_loss += self.ce_loss(cls_id_pred, cls_id_target) / float(cls_id_target.nelement())

                    # cls_id_pred = self.classifier(id_head).contiguous()
                    # if self.opt.id_loss == 'focal':
                    #     reid_target_one_hot = cls_id_pred.new_zeros((cls_id_head.size(0), self.nID)).scatter_(1,
                    #                                                                                 cls_id_target.long().view(
                    #                                                                                     -1, 1), 1)
                    #     reid_loss += sigmoid_focal_loss_jit(cls_id_pred, reid_target_one_hot,
                    #                                     alpha=0.25, gamma=2.0, reduction="sum"
                    #                                     ) / cls_id_pred.size(0)
                    # else:
                    #     reid_loss += self.ce_loss(cls_id_pred, cls_id_target)

        det_loss = opt.hm_weight * hm_loss + opt.wh_weight * wh_loss + opt.off_weight * off_loss

        if opt.multi_loss == 'uncertainty':
            loss = torch.exp(-self.s_det) * det_loss + torch.exp(-self.s_id) * reid_loss + (self.s_det + self.s_id)
            loss *= 0.5
        else:
            loss = det_loss + 0.1 * reid_loss

        loss_stats = {'loss': loss, 'hm_loss': hm_loss,
                      'wh_loss': wh_loss, 'off_loss': off_loss, 'id_loss': reid_loss}
        return loss, loss_stats


class MotTrainer(BaseTrainer):
    def __init__(self, opt, model, optimizer=None):
        super(MotTrainer, self).__init__(opt, model, optimizer=optimizer)

    def _get_losses(self, opt):
        loss_states = ['loss', 'hm_loss', 'wh_loss', 'off_loss', 'id_loss']
        # loss = MotLoss(opt)
        loss = McMotLoss(opt)  # multi-class multi-object tracking loss
        return loss_states, loss


    def save_result(self, output, batch, results):
        reg = output['reg'] if self.opt.reg_offset else None
        dets = mot_decode(
            output['hm'], output['wh'], reg=reg,
            cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.K)
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
        dets_out = ctdet_post_process(
            dets.copy(), batch['meta']['c'].cpu().numpy(),
            batch['meta']['s'].cpu().numpy(),
            output['hm'].shape[2], output['hm'].shape[3], output['hm'].shape[1])
        results[batch['meta']['img_id'].cpu().numpy()[0]] = dets_out[0]
