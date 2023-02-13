from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os


class opts(object):
  def __init__(self):
    self.parser = argparse.ArgumentParser()
    
    # basic experiment setting
    self.parser.add_argument('task', default='mot', help='mot')
    self.parser.add_argument('--dataset', default='jde', help='jde')
    self.parser.add_argument('--exp_id', default='default')
    self.parser.add_argument('--test', action='store_true')
    self.parser.add_argument('--load_model', default='',
                             help='path to pretrained model')
    self.parser.add_argument('--resume', action='store_true',
                             help='resume an experiment. '
                                  'Reloaded the optimizer parameter and '
                                  'set load_model to model_last.pth '
                                  'in the exp dir if load_model is empty.')
    self.parser.add_argument('--store_opt', action='store_true',
                             help='Whether the opts and call cmd shall be saved in the output during inference.')

    # system
    self.parser.add_argument('--gpus', default='0',
                             help='-1 for CPU, use comma for multiple gpus')
    self.parser.add_argument('--num_workers', type=int, default=8,
                             help='dataloader threads. 0 for single-thread.')
    self.parser.add_argument('--not_cuda_benchmark', action='store_true',
                             help='disable when the input size is not fixed.')
    self.parser.add_argument('--seed', type=int, default=317, 
                             help='Setting seed for all random generators in use.')

    # log
    self.parser.add_argument('--print_iter', type=int, default=0, 
                             help='disable progress bar and print to screen.')
    self.parser.add_argument('--hide_data_time', action='store_true',
                             help='not display time during training.')
    self.parser.add_argument('--metric', default='loss', 
                             help='main metric to save best model')
    # FIXME currently unused
    self.parser.add_argument('--vis_thresh', type=float, default=0.5,
                             help='visualization threshold.')
    
    # model
    self.parser.add_argument('--arch', default='dla_34', 
                             help='model architecture. Currently tested'
                                  'resdcn_34 | resdcn_50 | resfpndcn_34 |'
                                  'dla_34 | hrnet_18')
    self.parser.add_argument('--head_conv', type=int, default=-1,
                             help='conv layer channels for output head'
                                  '0 for no conv layer'
                                  '-1 for default setting: '
                                  '256 for resnets and 256 for dla.')
    self.parser.add_argument('--down_ratio', type=int, default=4,
                             help='output stride. Currently only supports 4.')
    self.parser.add_argument('--reid_dim', type=int, default=128,
                             help='feature dim for reid')
    
    # additional heads
    self.parser.add_argument('--gc_dim', type=int, default=128,
                             help='feature dim for gc')

    # input
    self.parser.add_argument('--input_h', type=int, default=608, 
                             help='input height.')
    self.parser.add_argument('--input_w', type=int, default=1088,
                             help='input width.')
    self.parser.add_argument('--ltrb', default=False,
                             help='regress left, top, right, bottom of bbox')
    
    # train
    self.parser.add_argument('--lr', type=float, default=1e-4,
                             help='learning rate for batch size 12.')
    self.parser.add_argument('--lr_step', type=str, default='20',
                             help='drop learning rate by 10.')
    self.parser.add_argument('--num_epochs', type=int, default=30,
                             help='total training epochs.')
    self.parser.add_argument('--batch_size', type=int, default=12,
                             help='batch size')
    self.parser.add_argument('--master_batch_size', type=int, default=-1,
                             help='batch size on the master gpu.')
    self.parser.add_argument('--num_iters', type=int, default=-1,
                             help='default: #samples / batch_size.')
    self.parser.add_argument('--val_intervals', type=int, default=5,
                             help='number of epochs to run validation.')
    self.parser.add_argument('--trainval', action='store_true',
                             help='include validation in training and '
                                  'test on test set')
    self.parser.add_argument('--save_all', action='store_true',
                             help='save more than just last model to disk.')

    # test
    self.parser.add_argument('--K', type=int, default=50,
                             help='max number of output objects.') 
    self.parser.add_argument('--not_prefetch_test', action='store_true',
                             help='not use parallal data pre-processing.')

    # output manipulation
    self.parser.add_argument('--line_thickness', type=int, default=1,
                             help='manipulate thickness of bb lines during inference.')
    self.parser.add_argument('--id_inline', action='store_true',
                             help='Wheter the tracking ID shall be printed next to the class name or above.')
    self.parser.add_argument('--debug_info', action='store_true',
                             help='Wheter scores of gc-labels shall be printed in top right corner.')

    # tracking
    self.parser.add_argument('--conf_thres', type=float, default=0.02, help='confidence threshold for tracking')
    self.parser.add_argument('--proportion_iou', type=float, default=0.5, help='which proportion should iou similarity get over cosine_similarity of ReID features; sim = proportion_iou * iou_sim + (1-proportion_iou) * emb_sim')
    self.parser.add_argument('--emb_sim_thres', type=float, default=0.6, help='embedding similarity threshold of new detection with detections from prior frames')
    self.parser.add_argument('--iou_sim_thres', type=float, default=0.5, help='iou similarity threshold of new detection with detections from prior frames')
    self.parser.add_argument('--det_thres', type=float, default=0.7, help='thresh for initializing new track')
    self.parser.add_argument('--new_overlap_thres', type=float, default=0.7, help='if current bb is overlapping more than this threshold with existing bb, no new track is started')    # are we using non-max-supression?
    self.parser.add_argument('--track_buffer', type=int, default=3, help='tracking buffer, in seconds how long a track should be kept active after last detection.')
    self.parser.add_argument('--min-box-area', type=float, default=100, help='filter out tiny boxes')

    # I/O-things
    self.parser.add_argument('--input_video', type=str,
                             default='../videos/MOT16-03.mp4',
                             help='path to the input video')
    self.parser.add_argument('--output_format', type=str, default='video', help='video or text')
    self.parser.add_argument('--output_root', type=str, default='../demos', help='expected output root path')
    self.parser.add_argument('--output_name', type=str, default='test', help='output video name')

    # mot
    self.parser.add_argument('--data_cfg', type=str,
                             default='../src/lib/cfg/data.json',
                             help='load data from cfg')
    self.parser.add_argument('--data_dir', type=str, default='../local_datasets/', help='set path to a dataset you want to use')

    # multi-class
    self.parser.add_argument('--reid_cls_names',
                                default='monkey,patch,kong,branch,XBI',
                                help='Define the names for the tracked classes.')

    # gc head (general classification)
    self.parser.add_argument('--use_gc', action='store_true',
                                help='Enable training / inference for general classification head.')
    self.parser.add_argument('--gc_cls_names',
                                default='walking,sitting,standing2legs,standing4legs,NiS',
                                help='Define the names for the possible classification targets.')
    self.parser.add_argument('--clsID4GC',
                                 default=0,
                                 help="Object class ID for which the classification shall be computed.")
    self.parser.add_argument('--train_only_gc', action='store_true',
                              help='Freeze all weights except general classification head + classification.')

    # loss
    self.parser.add_argument('--mse_loss', action='store_true',
                             help='use mse loss or focal loss to train '
                                  'keypoint heatmaps.')
    self.parser.add_argument('--reg_loss', default='l1',
                             help='regression loss: sl1 | l1 | l2')
    self.parser.add_argument('--hm_weight', type=float, default=1,
                             help='loss weight for keypoint heatmaps.')
    self.parser.add_argument('--off_weight', type=float, default=1,
                             help='loss weight for keypoint local offsets.')
    self.parser.add_argument('--wh_weight', type=float, default=0.1,
                             help='loss weight for bounding box size.')
    self.parser.add_argument('--id_loss', default='ce',
                             help='reid loss: ce | focal')
    self.parser.add_argument('--id_weight', type=float, default=1,
                             help='loss weight for id')
    self.parser.add_argument('--multi_loss', default='uncertainty', help='multi_task loss: uncertainty | fix')
    self.parser.add_argument('--gc_loss', default='CrEn',
                             help='gc loss: CrEn | none')
    
    # additional head settings
    self.parser.add_argument('--norm_wh', action='store_true',
                             help='L1(\hat(y) / y, 1) or L1(\hat(y), y)')
    self.parser.add_argument('--dense_wh', action='store_true',
                             help='apply weighted regression near center or '
                                  'just apply regression on center point.')
    self.parser.add_argument('--cat_spec_wh', action='store_true',
                             help='category specific bounding box size.')
    self.parser.add_argument('--not_reg_offset', action='store_true',
                             help='not regress local offset.')
    
    
    self.parser.add_argument('-f')


  def parse(self, args=''):
    if args == '':
      opt = self.parser.parse_args()
    else:
      opt = self.parser.parse_args(args)
    opt.gpus_str = opt.gpus
    opt.gpus = [int(gpu) for gpu in opt.gpus.split(',')]
    opt.lr_step = [int(i) for i in opt.lr_step.split(',')]

    opt.reg_offset = not opt.not_reg_offset

    if opt.head_conv == -1: # init default head_conv
      opt.head_conv = 256 if 'dla' in opt.arch else 256

    opt.pad = 31
    opt.num_stacks = 1

    if opt.master_batch_size == -1:
      opt.master_batch_size = opt.batch_size // len(opt.gpus)
    rest_batch_size = (opt.batch_size - opt.master_batch_size)
    opt.chunk_sizes = [opt.master_batch_size]
    for i in range(len(opt.gpus) - 1):
      slave_chunk_size = rest_batch_size // (len(opt.gpus) - 1)
      if i < rest_batch_size % (len(opt.gpus) - 1):
        slave_chunk_size += 1
      opt.chunk_sizes.append(slave_chunk_size)
    print('training chunk_sizes:', opt.chunk_sizes)

    opt.root_dir = os.path.join(os.path.dirname(__file__), '..', '..')
    opt.exp_dir = os.path.join(opt.root_dir, 'exp', opt.task)
    opt.save_dir = os.path.join(opt.exp_dir, opt.exp_id)
    opt.debug_dir = os.path.join(opt.save_dir, 'debug')
    opt.tb_dir = os.path.join(opt.save_dir, 'tensor_board')
    print('The output will be saved to ', opt.save_dir)
    
    if opt.resume and opt.load_model == '':
      model_path = opt.save_dir[:-4] if opt.save_dir.endswith('TEST') \
                  else opt.save_dir
      opt.load_model = os.path.join(model_path, 'model_last.pth')
    return opt

  def update_dataset_info_and_set_heads(self, opt, dataset):
    input_h, input_w = dataset.default_resolution
    opt.mean, opt.std = dataset.mean, dataset.std
    opt.num_classes = dataset.num_classes
    opt.class_names = dataset.class_names
    if opt.use_gc:
      opt.gc_cls_names = dataset.gc_cls_names
      opt.num_gc_cls = dataset.num_gc_cls

    opt.reid_cls_ids = list(range(opt.num_classes))

    opt.output_h = opt.input_h // opt.down_ratio
    opt.output_w = opt.input_w // opt.down_ratio

    if opt.task == 'mot':
      opt.heads = {'hm': opt.num_classes,
                    'wh': 2 if not opt.cat_spec_wh else 2 * opt.num_classes,# 'wh': 2 if not opt.ltrb else 4,
                    'id': opt.reid_dim}
    
    else:
      assert 0, 'task not defined!'
    
    if opt.use_gc:
      opt.heads.update({'gc': opt.gc_dim})
    if opt.reg_offset:
      opt.heads.update({'reg': 2})
    if opt.id_weight > 0:
      opt.nID_dict = dataset.nID_dict

    
    print('heads', opt.heads)
    return opt

  def init(self, args=''):
    opt = self.parse(args)
    default_dataset_info = {
            'mot': {'default_resolution': [opt.input_h, opt.input_w],
                    'num_classes': len(opt.reid_cls_names.split(',')),
                    'class_names': opt.reid_cls_names.split(','),
                    'mean': [0.408, 0.447, 0.470],
                    'std': [0.289, 0.274, 0.278],
                    'dataset': 'jde',
                    'nID_dict': {}},
    }
    if opt.use_gc:
      default_dataset_info['mot'].update({
                      'num_gc_cls': len(opt.gc_cls_names.split(',')),
                      'gc_cls_names': opt.gc_cls_names.split(',')
      })

    class Struct:
      def __init__(self, entries):
        for k, v in entries.items():
          self.__setattr__(k, v)
    h, w = default_dataset_info[opt.task]['default_resolution']
    opt.img_size = (w, h)
    print('Net input image size: {:d}×{:d}'.format(w, h))
    dataset = Struct(default_dataset_info[opt.task])
    opt.dataset = dataset.dataset
    opt = self.update_dataset_info_and_set_heads(opt, dataset)
    return opt
