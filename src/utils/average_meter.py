from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import time


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
            self.avg = self.sum / self.count


class TimeMeter(object):

    def __init__(self, total_epochs, iters_per_epoch):
        self.total_epochs = total_epochs
        self.iters_per_epoch = iters_per_epoch

        self.start_time = time.time()
        self.left_time = 0
        self.total_time = 0

        self.done_iters = 0
        self.left_iters = 0

    def update(self, epoch, now_iter_id):
        self.done_iters += 1
        self.total_time = time.time() - self.start_time
        self.left_iters = (self.total_epochs - epoch) * self.iters_per_epoch + \
            (self.iters_per_epoch-now_iter_id - 1)
        self.left_time = (self.left_iters / self.done_iters) * \
            self.total_time/3600
