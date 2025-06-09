# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Experimental modules
"""

# demo.py

import sys, os, importlib.util, types

# ────── 1. “虚拟”一个顶级包 models ──────
#    先造一个空模块，让 Python 认为：models 这个包已经在 sys.modules 里了
fake_models_pkg = types.ModuleType("models")
# （__path__ 可以留空，或者设为 []，常见做法是给空列表）
fake_models_pkg.__path__ = []  
sys.modules["models"] = fake_models_pkg

# ────── 2. 把 YOLOv5 源码的 yolo.py、common.py 以“models.yolo”、“models.common”名义加载进来 ──────
#    请根据你项目的实际结构，修改下面这两个路径

# 实际存放 yolo.py 的绝对路径
yolo_py_path = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "yolo.py"
    )
)
# 实际存放 common.py 的绝对路径
common_py_path = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "common.py"
    )
)

# ─── 2.1 加载 yolo.py 到 sys.modules["models.yolo"] ───
spec_yolo = importlib.util.spec_from_file_location("models.yolo", yolo_py_path)
mod_yolo = importlib.util.module_from_spec(spec_yolo)
spec_yolo.loader.exec_module(mod_yolo)
sys.modules["models.yolo"] = mod_yolo

# ─── 2.2 加载 common.py 到 sys.modules["models.common"] ───
spec_common = importlib.util.spec_from_file_location("models.common", common_py_path)
mod_common = importlib.util.module_from_spec(spec_common)
spec_common.loader.exec_module(mod_common)
sys.modules["models.common"] = mod_common

# 到这里为止：
#   sys.modules["models"] 存在（一个空的包），
#   sys.modules["models.yolo"]、sys.modules["models.common"] 分别指向 YOLOv5 里实际的 yolo.py、common.py


import math

import numpy as np
import torch
import torch.nn as nn

from demonstration.deep_sort.detector.YOLOv5.models.common import Conv
# from models.common import Conv
from demonstration.deep_sort.detector.YOLOv5.utils.downloads import attempt_download


class CrossConv(nn.Module):
    # Cross Convolution Downsample
    def __init__(self, c1, c2, k=3, s=1, g=1, e=1.0, shortcut=False):
        # ch_in, ch_out, kernel, stride, groups, expansion, shortcut
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, (1, k), (1, s))
        self.cv2 = Conv(c_, c2, (k, 1), (s, 1), g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class Sum(nn.Module):
    # Weighted sum of 2 or more layers https://arxiv.org/abs/1911.09070
    def __init__(self, n, weight=False):  # n: number of inputs
        super().__init__()
        self.weight = weight  # apply weights boolean
        self.iter = range(n - 1)  # iter object
        if weight:
            self.w = nn.Parameter(-torch.arange(1.0, n) / 2, requires_grad=True)  # layer weights

    def forward(self, x):
        y = x[0]  # no weight
        if self.weight:
            w = torch.sigmoid(self.w) * 2
            for i in self.iter:
                y = y + x[i + 1] * w[i]
        else:
            for i in self.iter:
                y = y + x[i + 1]
        return y


class MixConv2d(nn.Module):
    # Mixed Depth-wise Conv https://arxiv.org/abs/1907.09595
    def __init__(self, c1, c2, k=(1, 3), s=1, equal_ch=True):  # ch_in, ch_out, kernel, stride, ch_strategy
        super().__init__()
        n = len(k)  # number of convolutions
        if equal_ch:  # equal c_ per group
            i = torch.linspace(0, n - 1E-6, c2).floor()  # c2 indices
            c_ = [(i == g).sum() for g in range(n)]  # intermediate channels
        else:  # equal weight.numel() per group
            b = [c2] + [0] * n
            a = np.eye(n + 1, n, k=-1)
            a -= np.roll(a, 1, axis=1)
            a *= np.array(k) ** 2
            a[0] = 1
            c_ = np.linalg.lstsq(a, b, rcond=None)[0].round()  # solve for equal weight indices, ax = b

        self.m = nn.ModuleList(
            [nn.Conv2d(c1, int(c_), k, s, k // 2, groups=math.gcd(c1, int(c_)), bias=False) for k, c_ in zip(k, c_)])
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(torch.cat([m(x) for m in self.m], 1)))


class Ensemble(nn.ModuleList):
    # Ensemble of models
    def __init__(self):
        super().__init__()

    def forward(self, x, augment=False, profile=False, visualize=False):
        y = []
        for module in self:
            y.append(module(x, augment, profile, visualize)[0])
        # y = torch.stack(y).max(0)[0]  # max ensemble
        # y = torch.stack(y).mean(0)  # mean ensemble
        y = torch.cat(y, 1)  # nms ensemble
        return y, None  # inference, train output


def attempt_load(weights, map_location=None, inplace=True, fuse=True):
    from demonstration.deep_sort.detector.YOLOv5.models.yolo import Detect, Model
    # .yolo import Detect, Model

    # Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a
    model = Ensemble()
    # print(model)
    for w in weights if isinstance(weights, list) else [weights]:
        # ckpt = torch.load(attempt_download(w), map_location=map_location)  # load
        file_name = attempt_download(w)
        ckpt = torch.load(file_name, map_location=map_location)
        # ckpt = torch.load(file_name)
        print(ckpt)
        if fuse:
            model.append(ckpt['ema' if ckpt.get('ema') else 'model'].float().fuse().eval())  # FP32 model
        else:
            model.append(ckpt['ema' if ckpt.get('ema') else 'model'].float().eval())  # without layer fuse

    # Compatibility updates
    for m in model.modules():
        if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, Detect, Model]:
            m.inplace = inplace  # pytorch 1.7.0 compatibility
            if type(m) is Detect:
                if not isinstance(m.anchor_grid, list):  # new Detect Layer compatibility
                    delattr(m, 'anchor_grid')
                    setattr(m, 'anchor_grid', [torch.zeros(1)] * m.nl)
        elif type(m) is Conv:
            m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility

    if len(model) == 1:
        return model[-1]  # return model
    else:
        print(f'Ensemble created with {weights}\n')
        for k in ['names']:
            setattr(model, k, getattr(model[-1], k))
        model.stride = model[torch.argmax(torch.tensor([m.stride.max() for m in model])).int()].stride  # max stride
        return model  # return ensemble
