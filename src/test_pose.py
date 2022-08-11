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

    valset_paths = data_config['val']
    dataset_root = data_config['root']
    f.close()
    transforms = T.Compose([T.ToTensor()])
    # TODO why fixed input image size? opt.input_wh in mcmot code

    valset = JointDataset(opt, dataset_root, valset_paths, (1088, 608), augment=False, transforms=transforms)

    opt = opts().update_dataset_info_and_set_heads(opt, valset)
    print(opt)

    logger = Logger(opt)

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')

    print('Creating model...')
    if opt.use_pose:
        model = create_model(opt.arch, opt.heads, opt.head_conv, num_classes=opt.num_classes, num_poses=opt.num_poses, cat_spec_wh=opt.cat_spec_wh, clsID4Pose=opt.clsID4Pose, conf_thres=opt.conf_thres)
    else:
        model = create_model(opt.arch, opt.heads, opt.head_conv, num_classes=opt.num_classes, num_poses=None, cat_spec_wh=opt.cat_spec_wh, clsID4Pose=None, conf_thres=opt.conf_thres)
   
    optimizer = torch.optim.Adam(model.parameters(), opt.lr)

    start_epoch = 0

    # Get dataloader

    val_loader = torch.utils.data.DataLoader(
        valset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True
    )

    print('Starting validation...')
    trainer = MotTrainer(opt, model, optimizer)
    trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)

    if opt.load_model != '':
        model, optimizer, start_epoch = load_model(
            model, opt.load_model, trainer.optimizer, opt.resume, opt.lr, opt.lr_step)

    from sklearn.metrics import confusion_matrix
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    gt = []
    pred = []
    
    for iter_id, batch in enumerate(val_loader):
        for k in batch:
            if k != 'meta':
                batch[k] = batch[k].to(device=opt.device, non_blocking=True)
        outputs = model(batch['input'])
        output = outputs[-1]
        gt.append(batch['pose'].cpu().detach().numpy())
        pred.append(np.argmax(output['pose'].cpu().detach().numpy()))
       
       
    class_names = ['walking', 'sitting', 'standing2legs', 'standing4legs', 'NiS']
     
    cmat = confusion_matrix(gt, pred)
    
    
    fig = plt.figure(figsize=(5, 5), dpi=150)
    # plt.imshow(cmat, cmap='plasma')
    # plt.colorbar()
    hm = sns.heatmap(cmat, annot=True, fmt='d', linewidths=.5, cmap='plasma', square=True, cbar_kws={'shrink':0.6})
    # plt.title('Confusion Matrix of Classification', fontsize=20)
    # plt.xlabel('Prediction', labelpad=10, fontsize=20)
    # plt.ylabel('Ground-Truth', labelpad=10, fontsize=20)
    # plt.xticks(np.arange(len(class_names)), class_names)
    # plt.xticks(rotation=25)
    # plt.yticks(np.arange(len(class_names)), class_names)
    # plt.tight_layout()
    hm.set(ylabel='Ground Truth', xlabel='Prediction', title='Confusion Matrix of Classification')
    hm.set_xticklabels(labels=class_names, rotation=30)
    hm.set_yticklabels(labels=class_names, rotation=0)
    # plt.show()
    fig = hm.get_figure()
    fig.tight_layout()
    fig.savefig('cmat_val.png')

if __name__ == '__main__':
    torch.cuda.set_device(0)
    opt = opts().parse()
    main(opt)
