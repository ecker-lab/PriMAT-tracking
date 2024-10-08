from __future__ import absolute_import

import time

import numpy as np
import torch
from models.data_parallel import DataParallel
from progress.bar import Bar
from sklearn.metrics import confusion_matrix
from utils.utils import AverageMeter


class ModleWithLoss(torch.nn.Module):
    def __init__(self, model, loss, roi=False, square_bboxes = False, move_px = 0, zoom_min = 1, zoom_max = 1):
        super(ModleWithLoss, self).__init__()
        self.model = model
        self.loss = loss
        self.roi = roi
        self.square_bboxes = square_bboxes
        self.move_px = move_px
        self.zoom_min = zoom_min
        self.zoom_max = zoom_max

    def forward(self, batch):
        outputs = self.model(batch["input"])
        #if "gc" in self.model.heads:
        if batch['gc'].numel() > 0:
            if self.roi:
                outputs[-1]["gc_pred"], _ = self.model.gc_lin(batch['input'], batch, square_bboxes=self.square_bboxes, 
                                                              move_px = self.move_px, zoom_min = self.zoom_min, zoom_max = self.zoom_max) 
                #outputs[-1]["gc_pred"] = self.model.gc_lin(outputs[-1]["gc"], batch)
            else:
                outputs[-1]["gc_pred"] = self.model.gc_lin(
                    outputs[-1]["gc"], batch["cls_id_map"], batch["gc"], batch["gc_ct"]
                )
        loss, loss_stats = self.loss(outputs, batch)
        return outputs[-1], loss, loss_stats


class BaseTrainer(object):
    def __init__(self, opt, model, optimizer=None):
        self.opt = opt
        self.optimizer = optimizer
        self.loss_stats, self.loss = self._get_losses(opt)
        self.model_with_loss = ModleWithLoss(model, self.loss, opt.gc_with_roi, opt.squared_bboxes, opt.move_px, opt.zoom_min, opt.zoom_max)
        self.optimizer.add_param_group({"params": self.loss.parameters()})

    def set_device(self, gpus, chunk_sizes, device):
        if len(gpus) > 1:
            self.model_with_loss = DataParallel(
                self.model_with_loss, device_ids=gpus, chunk_sizes=chunk_sizes
            ).to(device)
        else:
            self.model_with_loss = self.model_with_loss.to(device)

        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device=device, non_blocking=True)

    def run_epoch(self, phase, epoch, data_loader):
        model_with_loss = self.model_with_loss
        if phase == "train":
            model_with_loss.train()
        else:
            if len(self.opt.gpus) > 1:
                model_with_loss = self.model_with_loss.module
            model_with_loss.eval()
            torch.cuda.empty_cache()
            # added for val
            gt = []
            pred = []

        opt = self.opt
        results = {}
        data_time, batch_time = AverageMeter(), AverageMeter()
        avg_loss_stats = {l: AverageMeter() for l in self.loss_stats}
        num_iters = len(data_loader) if opt.num_iters < 0 else opt.num_iters
        bar = Bar("{}/{}".format(opt.task, opt.exp_id), max=num_iters)
        end = time.time()
        for iter_id, batch in enumerate(data_loader):

            # check if batch['gc'] not None
            # Freeze self.model_with_loss.model -> parameters depending on task
            if iter_id >= num_iters:
                break
            data_time.update(time.time() - end)

            for k in batch:
                if k != "meta":
                    batch[k] = batch[k].to(device=opt.device, non_blocking=True)

            output, loss, loss_stats = model_with_loss(batch)
            loss = loss.mean()

            if phase == "train":
                with torch.set_grad_enabled(True):
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
            # added for eval
            else:
                if "gc" in self.loss_stats:
                    gt.append(batch["gc"].cpu().detach().numpy())
                    pred.append(np.argmax(output["gc"].cpu().detach().numpy()))

            batch_time.update(time.time() - end)
            end = time.time()

            Bar.suffix = "{phase}: [{0}][{1}/{2}]|Tot: {total:} |ETA: {eta:} ".format(
                epoch,
                iter_id,
                num_iters,
                phase=phase,
                total=bar.elapsed_td,
                eta=bar.eta_td,
            )
            for l in avg_loss_stats:
                avg_loss_stats[l].update(
                    loss_stats[l].mean().item(), batch["input"].size(0)
                )
                Bar.suffix = Bar.suffix + "|{} {:.4f} ".format(l, avg_loss_stats[l].avg)
            if not opt.hide_data_time:
                Bar.suffix = (
                    Bar.suffix + "|Data {dt.val:.3f}s({dt.avg:.3f}s) "
                    "|Net {bt.avg:.3f}s".format(dt=data_time, bt=batch_time)
                )
            if opt.print_iter > 0:
                if iter_id % opt.print_iter == 0:
                    print("{}/{}| {}".format(opt.task, opt.exp_id, Bar.suffix))
            else:
                bar.next()

            if opt.test:
                self.save_result(output, batch, results)
            del output, loss, loss_stats, batch

        #model_with_loss.eval()

        bar.finish()
        ret = {k: v.avg for k, v in avg_loss_stats.items()}
        ret["time"] = bar.elapsed_td.total_seconds() / 60.0
        if not phase == "train" and "gc_loss" in self.loss_stats:
            return ret, results, confusion_matrix(gt, pred)
        return ret, results

    def debug(self, batch, output, iter_id):
        raise NotImplementedError

    def save_result(self, output, batch, results):
        raise NotImplementedError

    def _get_losses(self, opt):
        raise NotImplementedError

    def val(self, epoch, data_loader):
        ret, results, *cmat = self.run_epoch("val", epoch, data_loader)
        if cmat:
            return ret, results, cmat[0]
        return ret, results

    def train(self, epoch, data_loader):
        return self.run_epoch("train", epoch, data_loader)