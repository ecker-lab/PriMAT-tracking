from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os

import json
import torch
import torch.utils.data
from torchvision.transforms import transforms as T
from opts import opts
from models.model import create_model, load_model, save_model
from models.data_parallel import DataParallel
from logger import Logger
from datasets.jde import JointDataset
from trains.mot import MotTrainer


def main(opt):
    torch.manual_seed(opt.seed)
    torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test

    print('Setting up data...')
    f = open(opt.data_cfg)
    data_config = json.load(f)
    trainset_paths = data_config['train']
    # added for val
    valset_paths = data_config['val']
    dataset_root = data_config['root']
    f.close()
    transforms = T.Compose([T.ToTensor()])
    # TODO why fixed input image size? opt.input_wh in mcmot code
    dataset = JointDataset(opt, dataset_root, trainset_paths, (1088, 608), augment=True, transforms=transforms)
    # added for val
    if opt.trainval:
        valset = JointDataset(opt, dataset_root, valset_paths, (1088, 608), augment=False, transforms=transforms)

    opt = opts().update_dataset_info_and_set_heads(opt, dataset)
    print(opt)

    logger = Logger(opt)

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')

    print('Creating model...')
    model = create_model(opt.arch, opt.heads, opt.head_conv, num_classes=opt.num_classes, num_poses=opt.num_poses, cat_spec_wh=opt.cat_spec_wh, clsID4Pose=opt.clsID4Pose, conf_thres=opt.conf_thres)
    if opt.train_only_pose:
        for name, param in model.named_parameters():
            if 'mpc' in name or 'pose_classifier' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        # for param in model.parameters():
        #     param.requires_grad = False        
        # for param in model.mpc.parameters():
        #     param.requires_grad = True
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), opt.lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), opt.lr)

    start_epoch = 0

    # Get dataloader

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True
    )

    if opt.trainval:
        val_loader = torch.utils.data.DataLoader(
            valset,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.num_workers,
            pin_memory=True,
            drop_last=True
        )

    print('Starting training...')
    trainer = MotTrainer(opt, model, optimizer)
    trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)

    if opt.load_model != '':
        model, optimizer, start_epoch = load_model(
            model, opt.load_model, trainer.optimizer, opt.resume, opt.lr, opt.lr_step)

    logger.graph_summary(model)

    lr = opt.lr
    for epoch in range(start_epoch + 1, opt.num_epochs + 1):
        mark = epoch if opt.save_all else 'last'
        log_dict_train, _ = trainer.train(epoch, train_loader)
        logger.write('epoch: {} |'.format(epoch))
        for k, v in log_dict_train.items():
            logger.scalar_summary('train_{}'.format(k), v, epoch)
            logger.write('{} {:8f} | '.format(k, v))
        logger.scalar_summary('learn_rate', lr, epoch)
        if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(mark)),
                       epoch, model, optimizer)
        else:
            save_model(os.path.join(opt.save_dir, 'model_last.pth'),
                       epoch, model, optimizer)
        logger.write('\n')
        if epoch in opt.lr_step:
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),
                       epoch, model, optimizer)
            lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
            print('Drop LR to', lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        if epoch % 10 == 0:
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),
                       epoch, model, optimizer)
        if opt.trainval:
            if opt.val_intervals > 0 and opt.val_intervals % epoch == 0:
                log_dict_val, _, cmat = trainer.val(epoch, val_loader)
                for k, v in log_dict_val.items():
                    logger.scalar_summary('val_{}'.format(k), v, epoch)
                    
                    logger.write('{} {:8f} | '.format(k, v))
                logger.write('\n')
                logger.val_summary('val_cmat', cmat, epoch)
    logger.close()


if __name__ == '__main__':
    torch.cuda.set_device(0)
    opt = opts().parse()
    main(opt)
