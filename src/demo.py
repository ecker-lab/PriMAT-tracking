from __future__ import absolute_import

import _init_paths

import logging
import os
import os.path as osp

import datasets.dataset as datasets
from logger import save_opt
from opts import opts
from tracking_utils.log import logger
from tracking_utils.utils import init_seeds, mkdir_if_missing


from track import eval_seq

logger.setLevel(logging.INFO)


def demo(opt):
    init_seeds(opt.seed)
    
    result_root = opt.output_root if opt.output_root != '' else '.'
    mkdir_if_missing(result_root)

    if opt.store_opt:
        save_opt(opt)

    logger.info('Starting tracking...')
    dataloader = datasets.LoadVideo(opt.input_video, opt.img_size)
    video_name = os.path.splitext(os.path.basename(opt.input_video))[0]
    result_filename = os.path.join(result_root, 'results.txt')

    frame_dir = None if opt.output_format == 'text' else osp.join(result_root, 'frame')

    eval_seq(opt, dataloader, opt.task, result_filename,save_dir=frame_dir,
            show_image=False, frame_rate=dataloader.frame_rate,
            use_cuda=opt.gpus!=[-1], video_name=video_name)

    if opt.output_format == 'video':
        output_video_path = osp.join(result_root, opt.output_name + '.mp4')
        cmd_str = 'ffmpeg -framerate {} -y -f image2 -i {}/%05d.jpg -b:v 5000k -c:v libx264 -vf format=yuv420p {}'.format(
            dataloader.frame_rate, osp.join(result_root, 'frame'), output_video_path)
        os.system(cmd_str)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    opt = opts().init()
    demo(opt)
