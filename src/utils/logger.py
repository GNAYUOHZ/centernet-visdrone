from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import sys
import torch


class Logger(object):
    def __init__(self, opt, name):
        self.log = open(opt.save_dir + f'/{name}.txt', 'a')
        self.write(f'==> torch version: {torch.__version__}')
        self.write(f'==> cudnn version: {torch.backends.cudnn.version()}')
        self.write(f'==> Cmd: {str(sys.argv)}')
        self.write(f'==> Opt:')
        args = dict((name, getattr(opt, name))
                    for name in dir(opt) if not name.startswith('_'))
        for k, v in sorted(args.items()):
            self.write(f'  {k}: {v}')
    	
    def write(self, txt):
        time_str = time.strftime('%Y-%m-%d %H:%M:%S')
        self.log.write(f'{time_str}: {txt}\n')
        self.log.flush()
        print(f'{time_str}: {txt}')

    def close(self):
        self.log.close()
