import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.update import BasicUpdateBlock, SmallUpdateBlock
from models.extractor import BasicEncoder, SmallEncoder
from models.corr import CorrBlock, AlternateCorrBlock, LocalCorrPyramid
from core.utils import bilinear_sampler, coords_grid, upflow8
autocast = torch.cuda.amp.autocast

class LocalCorrPyramid(nn.Module):
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        super(LocalCorrPyramid, self).__init__()
        self.num_levels = num_levels
        self.radius = radius
        
        # 构建特征金字塔
        self.pyramid1 = self._build_pyramid(fmap1)
        self.pyramid2 = self._build_pyramid(fmap2)
        
    def _build_pyramid(self, fmap):
        # 创建特征金字塔
        pyramid = [fmap]
        for i in range(self.num_levels-1):
            pyramid.append(F.avg_pool2d(pyramid[-1], 2, stride=2))
        return pyramid
    
    def __call__(self, coords):
        # 调整坐标到每个金字塔层级
        coords_pyramid = []
        for i in range(self.num_levels):
            scale = 2 ** i
            coords_pyramid.append(coords / scale)
        
        # 在每个层级计算局部相关性
        corr_pyramid = []
        for i in range(self.num_levels):
            corr = self._compute_local_corr(
                self.pyramid1[i], self.pyramid2[i], coords_pyramid[i], self.radius
            )
            corr_pyramid.append(corr)
        
        return corr_pyramid
    
    def _compute_local_corr(self, fmap1, fmap2, coords, radius):
        """计算局部相关图"""
        B, D, H, W = fmap1.shape
        coords = coords.permute(0, 2, 3, 1)  # [B, H, W, 2]
        
        # 获取局部区域的坐标网格
        x_grid, y_grid = self._get_local_grids(coords, radius, H, W)
        
        # 从特征图2中采样局部区域
        fmap2_local = self._sample_features(fmap2, x_grid, y_grid)
        
        # 计算相关性
        fmap1 = fmap1.permute(0, 2, 3, 1)  # [B, H, W, D]
        corr = torch.matmul(fmap1.unsqueeze(4), fmap2_local.unsqueeze(5)).squeeze(-1)
        
        # 归一化
        corr = corr / torch.sqrt(torch.tensor(D, dtype=torch.float32))
        
        return corr.permute(0, 3, 1, 2)  # [B, (2r+1)^2, H, W]
    
    def _get_local_grids(self, coords, radius, H, W):
        """生成局部区域的坐标网格"""
        x, y = coords[..., 0], coords[..., 1]
        
        # 创建局部区域的偏移量
        offsets = torch.arange(-radius, radius+1, device=coords.device)
        x_offsets, y_offsets = torch.meshgrid(offsets, offsets, indexing='ij')
        x_offsets = x_offsets.reshape(-1)
        y_offsets = y_offsets.reshape(-1)
        
        # 计算局部区域的所有坐标
        x_grid = x.unsqueeze(-1) + x_offsets
        y_grid = y.unsqueeze(-1) + y_offsets
        
        # 归一化到[-1, 1]范围
        x_grid = 2 * x_grid / (W - 1) - 1
        y_grid = 2 * y_grid / (H - 1) - 1
        
        return x_grid, y_grid
    
    def _sample_features(self, fmap, x_grid, y_grid):
        """使用双线性插值从特征图中采样"""
        B, D, H, W = fmap.shape
        grid = torch.stack([x_grid, y_grid], dim=-1)  # [B, H, W, (2r+1)^2, 2]
        
        # 调整形状以适应grid_sample
        grid = grid.reshape(B, H * W * (2*self.radius+1)**2, 1, 2)
        fmap = fmap.unsqueeze(2).expand(-1, -1, H * W * (2*self.radius+1)**2, -1, -1)
        fmap = fmap.reshape(B, D, H * W, W)
        
        # 采样
        sampled = F.grid_sample(fmap, grid, mode='bilinear', padding_mode='border', align_corners=True)
        sampled = sampled.reshape(B, D, H, W, (2*self.radius+1)**2)
        
        return sampled.permute(0, 2, 3, 4, 1)  # [B, H, W, (2r+1)^2, D]

# 修改RAFT模型以集成局部相关金字塔
class ModifiedRAFT(nn.Module):
    def __init__(self, args):
        super(ModifiedRAFT, self).__init__()
        self.args = args

        if args.small:
            self.hidden_dim = hdim = 96
            self.context_dim = cdim = 64
            args.corr_levels = 4
            args.corr_radius = 3
        
        else:
            self.hidden_dim = hdim = 128
            self.context_dim = cdim = 128
            args.corr_levels = 4
            args.corr_radius = 4

        if 'dropout' not in self.args:
            self.args.dropout = 0

        if 'alternate_corr' not in self.args:
            self.args.alternate_corr = False

        # feature network, context network, and update block
        if args.small:
            print("small")
            self.fnet = SmallEncoder(output_dim=128, norm_fn='instance', dropout=args.dropout)        
            self.cnet = SmallEncoder(output_dim=hdim+cdim, norm_fn='none', dropout=args.dropout)
            # self.update_block = ModifiedSmallUpdateBlock(self.args, hidden_dim=hdim)

        else:
            self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=args.dropout)        
            self.cnet = BasicEncoder(output_dim=hdim+cdim, norm_fn='batch', dropout=args.dropout)
            self.update_block = ModifiedBasicUpdateBlock(self.args, hidden_dim=hdim)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//8, W//8, device=img.device)
        coords1 = coords_grid(N, H//8, W//8, device=img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8*H, 8*W)

    def forward(self, image1, image2, iters=12, flow_init=None, upsample=True, test_mode=False):
        """ Estimate optical flow between pair of frames """
        # 将图片从[0, 255]缩放到[-1, 1]
        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0
        # 将图片转换为内存连续的张量，以提高内存访问效率
        image1 = image1.contiguous()
        image2 = image2.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        # 特征提取器
        with autocast(enabled=self.args.mixed_precision):
            fmap1, fmap2 = self.fnet([image1, image2])        
        
        fmap1 = fmap1.float()
        fmap2 = fmap2.float()
        
        # 使用局部相关金字塔替代原有的相关模块
        corr_fn = LocalCorrPyramid(fmap1, fmap2, num_levels=self.args.corr_levels, radius=self.args.corr_radius)

        # run the context network
        with autocast(enabled=self.args.mixed_precision):
            cnet = self.cnet(image1)
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)

        coords0, coords1 = self.initialize_flow(image1)

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()
            corr_pyramid = corr_fn(coords1) # 获取多尺度相关图

            flow = coords1 - coords0
            with autocast(enabled=self.args.mixed_precision):
                net, up_mask, delta_flow = self.update_block(net, inp, corr_pyramid, flow)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)
            
            flow_predictions.append(flow_up)

        if test_mode:
            return coords1 - coords0, flow_up
            
        return flow_predictions

# 修改更新块以处理多尺度相关图
class ModifiedBasicUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dim):
        super(ModifiedBasicUpdateBlock, self).__init__()
        self.args = args
        self.encoder = CorrEncoder(args.corr_levels, args.corr_radius, hidden_dim)
        self.gru = GatedUpdateNetwork(hidden_dim)
        
    def forward(self, net, inp, corr_pyramid, flow):
        # 处理多尺度相关图
        corr_features = self.encoder(corr_pyramid)
        
        # 更新网络状态
        net, up_mask, delta_flow = self.gru(net, inp, corr_features, flow)
        
        return net, up_mask, delta_flow

# 新增相关编码器处理多尺度相关图
class CorrEncoder(nn.Module):
    def __init__(self, num_levels, radius, hidden_dim):
        super(CorrEncoder, self).__init__()
        self.num_levels = num_levels
        
        # 为每个金字塔层级创建编码器
        self.level_encoders = nn.ModuleList()
        for i in range(num_levels):
            input_dim = (2 * radius + 1) ** 2
            self.level_encoders.append(nn.Sequential(
                nn.Conv2d(input_dim, hidden_dim//2, 1, padding=0),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim//2, hidden_dim//2, 3, padding=1),
                nn.ReLU(inplace=True)
            ))
        
        # 融合多层特征
        self.fusion = nn.Sequential(
            nn.Conv2d(hidden_dim//2 * num_levels, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        )
    
    def forward(self, corr_pyramid):
        # 编码每个层级的相关图
        encoded_levels = []
        for i in range(self.num_levels):
            encoded = self.level_encoders[i](corr_pyramid[i])
            # 上采样到相同尺寸
            if i > 0:
                encoded = F.interpolate(encoded, scale_factor=2**i, mode='bilinear', align_corners=True)
            encoded_levels.append(encoded)
        
        # 融合所有层级的特征
        fused = torch.cat(encoded_levels, dim=1)
        fused = self.fusion(fused)
        
        return fused