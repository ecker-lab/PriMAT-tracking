from __future__ import absolute_import

import math

import torch
import torch.nn.functional as F
from models.losses import (
    FocalLoss,
    NormRegL1Loss,
    RegL1Loss,
    RegLoss,
    RegWeightedL1Loss,
)
from models.utils import _sigmoid

from models.decode import mot_decode
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
        if opt.use_gc:
            self.crit_gc = torch.nn.CrossEntropyLoss(reduction='sum')
            
        self.opt = opt

        self.emb_dim = opt.reid_dim

        self.nID_dict = opt.nID_dict

        self.classifiers = torch.nn.ModuleDict()
        for cls_id, nID in self.nID_dict.items():
            self.classifiers[str(cls_id)] = torch.nn.Linear(self.emb_dim, nID)
        
        self.id_loss = torch.nn.CrossEntropyLoss(ignore_index=-1)
        if opt.id_loss == 'focal':
            for cls_id, nID in self.nID_dict.items():
                torch.nn.init.normal_(self.classifiers[str(cls_id)].weight, std=0.01)
                prior_prob = 0.01
                bias_value = -math.log((1 - prior_prob) / prior_prob)
                torch.nn.init.constant_(self.classifiers[str(cls_id)].bias, bias_value)

        self.emb_scale_dict = dict()
        for cls_id, nID in self.nID_dict.items():
            self.emb_scale_dict[cls_id] = math.sqrt(2) * math.log(nID - 1)
        # scale factor of reid loss
        self.s_id = torch.nn.Parameter(-1.05 * torch.ones(1))
        # scale factor of detection loss
        self.s_det = torch.nn.Parameter(-1.85 * torch.ones(1))


    def forward(self, outputs, batch):
        opt = self.opt

        hm_loss, wh_loss, off_loss, reid_loss = 0, 0, 0, 0
        if opt.use_gc:
            gc_loss = 0

        for s in range(opt.num_stacks):
            # ----- Detection loss
            output = outputs[s]
            if not opt.mse_loss:
                output['hm'] = _sigmoid(output['hm'])

            # --- heat-map loss
            hm_loss += self.crit(output['hm'], batch['hm']) / opt.num_stacks
            
            # --- box width and height loss
            if opt.wh_weight > 0:
                # TODO rename reg_mask to something more useful! where? -> jde.py mot.py, multitracker.py
                wh_loss += self.crit_wh(
                    output['wh'], batch['reg_mask'],
                    batch['ind'], batch['wh']) / opt.num_stacks

            # --- bbox center offset loss
            if opt.reg_offset and opt.off_weight > 0:
                off_loss += self.crit_reg(output['reg'], batch['reg_mask'],
                                          batch['ind'], batch['reg']) / opt.num_stacks
           
            # ----- ReID loss: only process the class requiring ReID
            if opt.id_weight > 0:
                cls_id_map = batch['cls_id_map']
                for cls_id, id_num in self.nID_dict.items():
                    inds = torch.where(cls_id_map == cls_id)
                    # skip not relevant classes
                    if inds[0].shape[0] == 0:
                        continue
                    cls_id_head = output['id'][inds[0], :, inds[2], inds[3]]
                    cls_id_head = self.emb_scale_dict[cls_id] * F.normalize(cls_id_head)
                    cls_id_target = batch['cls_tr_ids'][inds[0], cls_id, inds[2], inds[3]]

                    cls_id_pred = self.classifiers[str(cls_id)].forward(cls_id_head).contiguous()

                    reid_loss += self.id_loss(cls_id_pred, cls_id_target) / float(cls_id_target.nelement())

            if opt.use_gc:
                gc_loss += self.crit_gc(output['gc_pred'], batch['gc']) / batch['gc'].numel()


        det_loss = opt.hm_weight * hm_loss + opt.wh_weight * wh_loss + opt.off_weight * off_loss


        
        if opt.multi_loss == 'uncertainty':
            # FIXME loss scalings -- gc loss way to huge?????
            loss = torch.exp(-self.s_det) * det_loss + torch.exp(-self.s_id) * reid_loss + (self.s_det + self.s_id)
            # FIXME why only take half loss?????? TIMO???? HEEEELP MEEEEEE!!!!
            loss *= 0.5
        else:
            loss = det_loss + 0.1 * reid_loss


        loss_stats = {'loss': loss, 'hm_loss': hm_loss,
                    'wh_loss': wh_loss, 'off_loss': off_loss, 'id_loss': reid_loss}

        if opt.use_gc:
            loss += 0.5 * gc_loss

            loss_stats.update({'loss': loss, 'gc_loss': gc_loss})
        
        return loss, loss_stats


class MotTrainer(BaseTrainer):
    def __init__(self, opt, model, optimizer=None):
        super(MotTrainer, self).__init__(opt, model, optimizer=optimizer)

    def _get_losses(self, opt):
        if 'gc' in opt.heads:
            loss_states = ['loss', 'hm_loss', 'wh_loss', 'off_loss', 'id_loss', 'gc_loss']
        else:
            loss_states = ['loss', 'hm_loss', 'wh_loss', 'off_loss', 'id_loss']
        loss = MotLoss(opt)
        return loss_states, loss

    # TODO might have to include pose here as well?????
    def save_result(self, output, batch, results):
        reg = output['reg'] if self.opt.reg_offset else None
        dets = mot_decode(
            output['hm'], output['wh'], reg=reg, num_classes=self.opt.num_classes,
            cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.K)
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
        dets_out = ctdet_post_process(
            dets.copy(), batch['meta']['c'].cpu().numpy(),
            batch['meta']['s'].cpu().numpy(),
            output['hm'].shape[2], output['hm'].shape[3], output['hm'].shape[1])
        results[batch['meta']['img_id'].cpu().numpy()[0]] = dets_out[0]
