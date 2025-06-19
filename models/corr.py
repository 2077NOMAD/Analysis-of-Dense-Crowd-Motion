import torch
import torch.nn.functional as F
from core.utils import bilinear_sampler, coords_grid
import torch.nn as nn

try:
    import alt_cuda_corr
except:
    # alt_cuda_corr is not compiled
    pass


class CorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []

        # all pairs correlation
        corr = CorrBlock.corr(fmap1, fmap2)

        batch, h1, w1, dim, h2, w2 = corr.shape
        corr = corr.reshape(batch*h1*w1, dim, h2, w2)
        
        self.corr_pyramid.append(corr)
        for i in range(self.num_levels-1):
            corr = F.avg_pool2d(corr, 2, stride=2)
            self.corr_pyramid.append(corr)

    def __call__(self, coords):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            dx = torch.linspace(-r, r, 2*r+1, device=coords.device)
            dy = torch.linspace(-r, r, 2*r+1, device=coords.device)
            delta = torch.stack(torch.meshgrid(dy, dx), axis=-1)

            centroid_lvl = coords.reshape(batch*h1*w1, 1, 1, 2) / 2**i
            delta_lvl = delta.view(1, 2*r+1, 2*r+1, 2)
            coords_lvl = centroid_lvl + delta_lvl

            corr = bilinear_sampler(corr, coords_lvl)
            corr = corr.view(batch, h1, w1, -1)
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()

    @staticmethod
    def corr(fmap1, fmap2):
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht*wd)
        fmap2 = fmap2.view(batch, dim, ht*wd) 
        
        corr = torch.matmul(fmap1.transpose(1,2), fmap2)
        corr = corr.view(batch, ht, wd, 1, ht, wd)
        return corr  / torch.sqrt(torch.tensor(dim).float())


class AlternateCorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius

        self.pyramid = [(fmap1, fmap2)]
        for i in range(self.num_levels):
            fmap1 = F.avg_pool2d(fmap1, 2, stride=2)
            fmap2 = F.avg_pool2d(fmap2, 2, stride=2)
            self.pyramid.append((fmap1, fmap2))

    def __call__(self, coords):
        coords = coords.permute(0, 2, 3, 1)
        B, H, W, _ = coords.shape
        dim = self.pyramid[0][0].shape[1]

        corr_list = []
        for i in range(self.num_levels):
            r = self.radius
            fmap1_i = self.pyramid[0][0].permute(0, 2, 3, 1).contiguous()
            fmap2_i = self.pyramid[i][1].permute(0, 2, 3, 1).contiguous()

            coords_i = (coords / 2**i).reshape(B, 1, H, W, 2).contiguous()
            corr, = alt_cuda_corr.forward(fmap1_i, fmap2_i, coords_i, r)
            corr_list.append(corr.squeeze(1))

        corr = torch.stack(corr_list, dim=1)
        corr = corr.reshape(B, -1, H, W)
        return corr / torch.sqrt(torch.tensor(dim).float())


class LocalCorrPyramid(nn.Module):
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        super(LocalCorrPyramid, self).__init__()
        self.num_levels = num_levels
        self.radius = radius

        # 构建特征金字塔
        self.pyramid1 = self._build_pyramid(fmap1)
        self.pyramid2 = self._build_pyramid(fmap2)
        print(f"Feature pyramid built with {num_levels} levels and radius {radius}")

    def _build_pyramid(self, fmap):
        # 创建特征金字塔
        pyramid = [fmap]
        for i in range(self.num_levels - 1):
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
        offsets = torch.arange(-radius, radius + 1, device=coords.device)
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
        grid = grid.reshape(B, H * W * (2 * self.radius + 1) ** 2, 1, 2)
        fmap = fmap.unsqueeze(2).expand(-1, -1, H * W * (2 * self.radius + 1) ** 2, -1, -1)
        fmap = fmap.reshape(B, D, H * W, W)

        # 采样
        sampled = F.grid_sample(fmap, grid, mode='bilinear', padding_mode='border', align_corners=True)
        sampled = sampled.reshape(B, D, H, W, (2 * self.radius + 1) ** 2)

        return sampled.permute(0, 2, 3, 4, 1)  # [B, H, W, (2r+1)^2, D]
    
class LocalCorrPyramid:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []

        # 计算不同尺度下的相关性
        for i in range(num_levels):
            if i > 0:
                fmap1 = F.avg_pool2d(fmap1, 2, stride=2)
                fmap2 = F.avg_pool2d(fmap2, 2, stride=2)

            corr = self.compute_corr(fmap1, fmap2)
            self.corr_pyramid.append(corr)

    def __call__(self, coords):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            dx = torch.linspace(-r, r, 2*r+1, device=coords.device)
            dy = torch.linspace(-r, r, 2*r+1, device=coords.device)
            delta = torch.stack(torch.meshgrid(dy, dx), axis=-1)

            centroid_lvl = coords.reshape(batch*h1*w1, 1, 1, 2) / 2**i
            delta_lvl = delta.view(1, 2*r+1, 2*r+1, 2)
            coords_lvl = centroid_lvl + delta_lvl

            corr = bilinear_sampler(corr, coords_lvl)
            corr = corr.view(batch, h1, w1, -1)
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()

    def compute_corr(self, fmap1, fmap2):
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht*wd)
        fmap2 = fmap2.view(batch, dim, ht*wd)

        corr = torch.matmul(fmap1.transpose(1,2), fmap2)
        corr = corr.view(batch, ht, wd, 1, ht, wd)
        return corr / torch.sqrt(torch.tensor(dim).float())
