from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from .utils.decode import _transpose_and_gather_feat


class CtdetLoss(torch.nn.Module):
    def __init__(self, opt):
        super(CtdetLoss, self).__init__()
        self.crit_hm = FocalLoss()
        self.crit_reg = RegL1Loss()
        self.crit_wh = RegL1Loss()
        self.opt = opt

    def forward(self, output, batch):
        hm_loss, wh_loss, off_loss = 0, 0, 0

        output['hm'] = torch.clamp(output['hm'].sigmoid_(), 1e-4, 1-1e-4)
        output['wh'] = _transpose_and_gather_feat(output['wh'], batch['ind']) # 2 256 2
        output['reg'] = _transpose_and_gather_feat(output['reg'], batch['ind'])

        # loss1
        hm_loss += self.crit_hm(output['hm'], batch['hm'])
        # loss2
        wh_loss += self.crit_wh(output['wh'], batch['reg_mask'], batch['wh'])
        # loss3
        off_loss += self.crit_reg(output['reg'],batch['reg_mask'], batch['reg'])
        # total loss
        loss = self.opt.hm_weight * hm_loss \
            + self.opt.wh_weight * wh_loss \
            + self.opt.off_weight * off_loss

        loss_stats = {
            'loss': loss,
            'hm_loss': hm_loss,
            'wh_loss': wh_loss,
            'off_loss': off_loss
        }
        return loss_stats


class FocalLoss(torch.nn.Module):
    '''nn.Module warpper for focal loss'''

    def __init__(self):
        super(FocalLoss, self).__init__()

    def forward(self, out, target):
        ''' Modified focal loss. Exactly the same as CornerNet.
            Runs faster and costs a little bit more memory
            Arguments:
                out (batch x c x h x w)
                target (batch x c x h x w)
        '''
        alpha = 2
        beta = 4

        pos_inds = target.eq(1).float()
        neg_inds = target.lt(1).float()

        pos_loss = torch.log(out) * torch.pow(1 - out, alpha) * pos_inds
        neg_loss = torch.log(1 - out) * torch.pow(out, alpha) * \
            torch.pow(1 - target, beta) * neg_inds

        num_pos = pos_inds.sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            return - neg_loss
        else:
            return - (pos_loss + neg_loss) / num_pos


class RegL1Loss(torch.nn.Module):
    def __init__(self):
        super(RegL1Loss, self).__init__()

    def forward(self, output, mask, target):
        mask = mask.unsqueeze(2).expand_as(output).float()  # 2,256,2
        loss = torch.nn.functional.l1_loss(
            output * mask, target * mask, size_average=False)
        loss = loss / (mask.sum() + 1e-4)
        return loss
