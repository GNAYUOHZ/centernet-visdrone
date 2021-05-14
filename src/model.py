from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from torch.cuda.amp import autocast
from .backbones.resnet import ResNet, BasicBlock_ResNet, Bottleneck_ResNet
from .backbones.res2net import Res2Net, Bottleneck_Res2Net


def _make_deconv_layer(inplanes, outplanes):
    layers = []
    layers.append(
        nn.ConvTranspose2d(
            in_channels=inplanes,
            out_channels=outplanes,
            kernel_size=4,
            stride=2,
            padding=1,
            output_padding=0,
            bias=False,
        )
    )
    layers.append(nn.BatchNorm2d(outplanes))
    layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)


class elementwise(nn.Module):
    def __init__(self, inc, outc):
        super(elementwise, self).__init__()
        self.conv1 = nn.Conv2d(inc, outc, kernel_size=1, stride=1, padding=0, bias=True)
        self.deconv = _make_deconv_layer(outc, outc)
        self.debn = nn.BatchNorm2d(outc)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, p, c):
        out = self.relu(self.debn(self.deconv(p)) + self.conv1(c))
        return out


class cat_conv(nn.Module):
    def __init__(self, inc, outc, catc):
        super(cat_conv, self).__init__()
        self.conv3 = nn.Conv2d(
            catc, outc, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.deconv = _make_deconv_layer(inc, inc)
        self.bn = nn.BatchNorm2d(outc)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, feat1, feat2):
        feat2 = self.deconv(feat2)
        cat_feat = torch.cat([feat1, feat2], dim=1)
        out = self.relu(self.bn(self.conv3(cat_feat)))
        return out


class fpn_deconv(nn.Module):
    def __init__(self, expansion):
        super(fpn_deconv, self).__init__()

        # Top layer
        self.toplayer = nn.Conv2d(512 * expansion, 256, 1, 1, 0)
        # Fpn
        self.elementwise1 = elementwise(512 * expansion // 2, 256)
        self.elementwise2 = elementwise(512 * expansion // 4, 256)
        self.elementwise3 = elementwise(512 * expansion // 8, 256)
        self.reduce_chan = nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1)

        self.cat_conv1_4 = cat_conv(64, 64, 128)

    def forward(self, c1_1, c2, c3, c4, c5):

        # Top-down
        p5_1 = self.toplayer(c5)  # 256 16 16
        p4_1 = self.elementwise1(p5_1, c4)  # 256 32 32
        p3_1 = self.elementwise2(p4_1, c3)  # 256 64 64
        p2_1 = self.elementwise3(p3_1, c2)  # 256 128 128
        last_feat = self.reduce_chan(p2_1)

        # down ratio 2
        last_feat = self.cat_conv1_4(c1_1, last_feat)  # [1, 64, 256, 256]
        return last_feat


class centernet(nn.Module):
    def __init__(self, spec, heads):
        super(centernet, self).__init__()

        arch = spec[0]
        block = spec[1]
        layers = spec[2]
        backbone = arch(block, layers)
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.heads = heads
        self.neck = fpn_deconv(block.expansion)

        # Head
        for head in sorted(self.heads):
            num_output = self.heads[head]
            fc = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, num_output, kernel_size=1, padding=0),
            )
            self.__setattr__(head, fc)

    @autocast()
    def forward(self, x):
        c1 = self.conv1(x)  # 3 
        c1 = self.bn1(c1)
        c1 = self.relu(c1)
        c1_1 = c1  # 64 
        c1 = self.maxpool(c1)  # 64 

        c2 = self.layer1(c1)  # 256 
        c3 = self.layer2(c2)  # 512 
        c4 = self.layer3(c3)  # 1024 
        c5 = self.layer4(c4)  # 2048 

        last_feat = self.neck(c1_1, c2, c3, c4, c5)

        ret = {}
        for head in self.heads:
            ret[head] = self.__getattr__(head)(last_feat)
        return ret

    def init_weights(self, url, pretrained=True):
        if pretrained:
            for m in self.modules():
                # if isinstance(m, nn.Conv2d):
                #     nn.init.normal_(m.weight, std=0.001)
                # elif isinstance(m, nn.BatchNorm2d):
                #     m.weight.data.fill_(1)
                #     m.bias.data.zero_()
                if isinstance(m, nn.ConvTranspose2d):
                    nn.init.normal_(m.weight, std=0.001)
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

            # print('=> init final conv weights from normal distribution')
            for head in self.heads:
                final_layer = self.__getattr__(head)
                for i, m in enumerate(final_layer.modules()):
                    if isinstance(m, nn.Conv2d):
                        if m.weight.shape[0] == self.heads[head]:
                            if "hm" in head:
                                nn.init.constant_(m.bias, -2.19)
                            else:
                                nn.init.normal_(m.weight, std=0.001)
                                nn.init.constant_(m.bias, 0)

            pretrained_state_dict = model_zoo.load_url(url)
            print("=> loading pretrained model {}".format(url))
            self.load_state_dict(pretrained_state_dict, strict=False)


model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
    "res2net50": "https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_v1b_26w_4s-3cf99910.pth",
    "res2net101": "https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net101_v1b_26w_4s-0812c246.pth",
}
resnet_spec = {
    "resnet18": (ResNet, BasicBlock_ResNet, [2, 2, 2, 2]),
    "resnet34": (ResNet, BasicBlock_ResNet, [3, 4, 6, 3]),
    "resnet50": (ResNet, Bottleneck_ResNet, [3, 4, 6, 3]),
    "resnet101": (ResNet, Bottleneck_ResNet, [3, 4, 23, 3]),
    "resnet152": (ResNet, Bottleneck_ResNet, [3, 8, 36, 3]),
    "res2net50": (Res2Net, Bottleneck_Res2Net, [3, 4, 6, 3]),
    "res2net101": (Res2Net, Bottleneck_Res2Net, [3, 4, 23, 3]),
}


def get_model(arch, heads):
    model = centernet(resnet_spec[arch], heads)
    model.init_weights(model_urls[arch], pretrained=True)
    return model


def load_model(model, model_path, optimizer=None, lr=None, lr_step=None):
    start_epoch = 0
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    print(f'loaded {model_path}, epoch {checkpoint["epoch"]}')
    state_dict_ = checkpoint["state_dict"]
    state_dict = {}

    # convert data_parallal to model
    for k in state_dict_:
        if k.startswith("module") and not k.startswith("module_list"):
            state_dict[k[7:]] = state_dict_[k]
        else:
            state_dict[k] = state_dict_[k]
    model_state_dict = model.state_dict()

    # check loaded parameters and created model parameters
    msg = (
        "If you see this, your model does not fully load the "
        + "pre-trained weight. Please make sure "
        + "you have correctly specified --arch xxx "
        + "or set the correct --num_classes for your own dataset."
    )
    for k in state_dict:
        if k in model_state_dict:
            if state_dict[k].shape != model_state_dict[k].shape:
                print(
                    f"Skip loading parameter {k}, required shape {model_state_dict[k].shape}, loaded shape {state_dict[k].shape}. {msg}"
                )
                state_dict[k] = model_state_dict[k]
        else:
            print(f"Drop parameter {k}. {msg}")
    for k in model_state_dict:
        if k not in state_dict:
            print(f"No param {k}. {msg}")
            state_dict[k] = model_state_dict[k]
    model.load_state_dict(state_dict, strict=False)

    # resume optimizer parameters
    if optimizer is not None:
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
            start_epoch = checkpoint["epoch"]
            start_lr = lr
            for step in lr_step:
                if start_epoch >= step:
                    start_lr *= 0.1
            for param_group in optimizer.param_groups:
                param_group["lr"] = start_lr
            print("Resumed optimizer with start lr", start_lr)
        else:
            print("No optimizer parameters in checkpoint.")
    if optimizer is not None:
        return model, optimizer, start_epoch
    else:
        return model


def save_model(path, epoch, model, optimizer=None):
    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    data = {"epoch": epoch, "state_dict": state_dict}
    if optimizer is not None:
        data["optimizer"] = optimizer.state_dict()
    torch.save(data, path)


if __name__ == "__main__":

    heads = {"hm": 10, "wh": 2, "reg": 2}
    net = get_model("resnet18", heads).cuda()

    x = torch.randn(2, 3, 512, 512).cuda()
    import time

    s = time.time()
    for i in range(10):
        y = net(x)
        print(time.time() - s)
        s = time.time()

    print(y["hm"].size())
