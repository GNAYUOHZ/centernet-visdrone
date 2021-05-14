from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch


def _gather_feat(feat, ind):

    dim = feat.size(2)  # c

    # ind 2,256 -> 2,256,1 -> 2,256,2
    # ind fmap中的序列数
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    # feat 2,76800,2-> 2,256,2
    feat = feat.gather(1, ind)

    return feat


# 获取 ground truth 中计算得到的对应中心点的值
def _transpose_and_gather_feat(feat, ind):

    # 有些tensor并不是占用一整块内存，而是由不同的数据块组成，
    # 而tensor的view()操作依赖于内存是整块的，
    # 这时只需要执行contiguous()这个函数，把tensor变成在内存中连续分布的形式。
    feat = feat.permute(0, 2, 3, 1).contiguous()  # batch,c,h,w -> batch,h,w,c
    # 将wh合并成一维
    feat = feat.view(feat.size(0), -1, feat.size(3))  # batch, w*h,c
    # ind 代表的是 ground truth 中设置的存在目标点的下角标
    feat = _gather_feat(feat, ind)
    return feat


def _nms(heat, kernel=3):
    hmax = torch.nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=(kernel - 1) // 2)
    keep = (hmax == heat).float()
    return heat * keep


def _topk(scores, K=40):
    batch, cat, height, width = scores.size()
    # 每个 class channel 统计最大值
    # topk_scores和topk_inds分别为每个batch每张heatmap（每个类别）中前K个最大的score和id。
    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)
    # torch.Size([1, 10, 500]) torch.Size([1, 10, 500])

    # 找到横纵坐标
    topk_ys = torch.true_divide(topk_inds, width).int().float()
    topk_xs = (topk_inds % width).int().float()

    # 在每个batch中取所有heatmap的前K个最大score以及id，不考虑类别的影响
    # topk_score：batch * K
    # topk_ind：batch * K 
    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    # 所有类别中找到最大值
    topk_clses = torch.true_divide(topk_ind, K).int()
    topk_inds = _gather_feat(topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


# 将 heatmap 转化成 bbox
def ctdet_decode(heat, wh, reg, K=100):
    batch = heat.size(0)

    # heat = torch.sigmoid(heat)
    # perform nms on heatmaps
    heat = _nms(heat)

    scores, inds, clses, ys, xs = _topk(heat, K=K)

    reg = _transpose_and_gather_feat(reg, inds)
    reg = reg.view(batch, K, 2)
    xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
    ys = ys.view(batch, K, 1) + reg[:, :, 1:2]

    wh = _transpose_and_gather_feat(wh, inds)
    wh = wh.view(batch, K, 2)

    clses = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)
    bboxes = torch.cat([xs - wh[..., 0:1] / 2,
                        ys - wh[..., 1:2] / 2,
                        xs + wh[..., 0:1] / 2,
                        ys + wh[..., 1:2] / 2], dim=2)
    detections = torch.cat([bboxes, scores, clses], dim=2)

    return detections

