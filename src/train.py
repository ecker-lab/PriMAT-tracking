from __future__ import absolute_import, print_function
import torch
import _init_paths

import os

import json

import torch.utils.data
from torchvision.transforms import transforms as T
from opts import opts
from models.model import create_model, load_model, save_model
from models.data_parallel import DataParallel
from logger import Logger
from datasets.dataset import JointDataset2, JointDataset
from trains.mot import MotTrainer
from tracking_utils.utils import init_seeds


def main(opt):
    init_seeds(opt.seed)
    torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test

    print('Setting up data...')
    f = open(opt.data_cfg)
    data_config = json.load(f)
    trainset_paths = data_config['train']
    # added for val
    if opt.trainval:
        valset_paths = data_config['val']
    dataset_root = data_config['root']
    

    try:
        gc_cnts_value = data_config['gc_cnts']
        opt.gc_lbl_cnts = data_config['gc_cnts']['train']
    except KeyError:
        opt.gc_lbl_cnts = False

    f.close()
    
    transforms = T.Compose([T.ToTensor()])
    # TODO why fixed input image size? why not size set in opts? opt.input_wh
    dataset = JointDataset2(opt, dataset_root, trainset_paths, (1088, 608), augment=True, transforms=transforms)
    # added for val
    if opt.trainval:
        valset = JointDataset(opt, dataset_root, valset_paths, (1088, 608), augment=False, transforms=transforms)

    opt = opts().update_dataset_info_and_set_heads(opt, dataset)
    print(opt)

    logger = Logger(opt)

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')

    print('Creating model...')
    if opt.use_gc:
        model = create_model(opt.arch, opt.heads, opt.head_conv, num_gc_cls=opt.num_gc_cls, clsID4GC=opt.clsID4GC, gc_with_roi=opt.gc_with_roi)
    else:
        model = create_model(opt.arch, opt.heads, opt.head_conv, num_gc_cls=None, clsID4GC=None, gc_with_roi=False)
    if opt.train_only_gc:
        for name, param in model.named_parameters():
            if 'gc' in name:
                param.requires_grad = True
                #print(name, "with gradient")
            else:
                param.requires_grad = False
                #print(name, "without gradient")

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), opt.lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), opt.lr)

    

    # logger.graph_summary(model)

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

    # images = next(iter(train_loader))['input']
    # logger.graph_summary(model, images)

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

    if opt.load_tracking_model != '':
        model, optimizer, start_epoch = load_model(
            model, opt.load_tracking_model, trainer.optimizer, opt.resume, opt.lr, opt.lr_step)

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
        # if epoch % 10 == 0:
        #     save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),
        #                epoch, model, optimizer)
        if opt.trainval:
            if epoch % opt.val_intervals == 0:
                if opt.use_gc:
                    log_dict_val, _, cmat = trainer.val(epoch, val_loader)
                else:
                    log_dict_val, _ = trainer.val(epoch, val_loader)
                for k, v in log_dict_val.items():
                    logger.scalar_summary('val_{}'.format(k), v, epoch)
                    
                    logger.write('{} {:8f} | '.format(k, v))
                logger.write('\n')
                if opt.use_gc:
                    logger.val_summary('val_cmat', cmat, epoch)
    logger.close()


if __name__ == '__main__':
    torch.cuda.set_device(0)
    opt = opts().parse()
    main(opt)
