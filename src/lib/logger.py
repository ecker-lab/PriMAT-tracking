from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
import os
import time
import sys
import torch
USE_TENSORBOARD = True
try:
  from torch.utils.tensorboard import SummaryWriter
  print('Using tensorboard')
except:
  USE_TENSORBOARD = False

class Logger(object):
  def __init__(self, opt):
    """Create a summary writer logging to log_dir."""
    if not os.path.exists(opt.save_dir):
      os.makedirs(opt.save_dir)
    if not os.path.exists(opt.debug_dir):
      os.makedirs(opt.debug_dir)
   
    time_str = time.strftime('%Y-%m-%d-%H-%M')

    args = dict((name, getattr(opt, name)) for name in dir(opt)
                if not name.startswith('_'))
    file_name = os.path.join(opt.save_dir, 'opt.txt')
    with open(file_name, 'wt') as opt_file:
      opt_file.write('==> torch version: {}\n'.format(torch.__version__))
      opt_file.write('==> cudnn version: {}\n'.format(
        torch.backends.cudnn.version()))
      opt_file.write('==> Cmd:\n')
      opt_file.write(str(sys.argv))
      opt_file.write('\n==> Opt:\n')
      for k, v in sorted(args.items()):
        opt_file.write('  %s: %s\n' % (str(k), str(v)))
          
    log_dir = opt.save_dir + '/logs_{}'.format(time_str)
    if USE_TENSORBOARD:
      self.writer = SummaryWriter(opt.tb_dir)
    if not os.path.exists(os.path.dirname(log_dir)):
      os.mkdir(os.path.dirname(log_dir))
    if not os.path.exists(log_dir):
      os.mkdir(log_dir)
    self.log = open(log_dir + '/log.txt', 'w')
    try:
      os.system('cp {}/opt.txt {}/'.format(opt.save_dir, log_dir))
    except:
      pass
    self.start_line = True

  def write(self, txt):
    if self.start_line:
      time_str = time.strftime('%Y-%m-%d-%H-%M')
      self.log.write('{}: {}'.format(time_str, txt))
    else:
      self.log.write(txt)  
    self.start_line = False
    if '\n' in txt:
      self.start_line = True
      self.log.flush()
  
  def close(self):
    self.log.close()
  
  # tensorboard code
  def scalar_summary(self, tag, value, step):
    """Log a scalar variable."""
    if USE_TENSORBOARD:
      self.writer.add_scalar(tag, value, step)

  def val_summary(self, tag, cmat, step):
    if USE_TENSORBOARD:
      import numpy as np
      class_names = ['walking', 'sitting', 'standing2legs', 'standing4legs', 'NiS']
      import matplotlib.pyplot as plt
      fig = plt.figure(figsize=(10, 10), dpi=300)
      plt.imshow(cmat, cmap='plasma')
      plt.colorbar()
      plt.title('Confusion Matrix of Classification', fontsize=20)
      plt.xlabel('Prediction', labelpad=10, fontsize=20)
      plt.ylabel('Ground-Truth', labelpad=10, fontsize=20)
      plt.xticks(np.arange(len(class_names)), class_names)
      plt.xticks(rotation=25)
      plt.yticks(np.arange(len(class_names)), class_names)
      plt.tight_layout()
      
      self.writer.add_figure(tag, fig, step)

