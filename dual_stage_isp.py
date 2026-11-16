"""Dual-Stage ISP: RestorationNet followed by EnhancementNet."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from arch_utils import RestorationNet, EncDec, ResidualBlockNoBN
from model_config import load_model_config


class GuideNN(nn.Module):
    def __init__(self, ci=3, co=1, kernel=1, layer=2, res=False, channel=16, scale=1):
        super().__init__()
        assert scale == 1, f'Not support scale={scale} yet!'
        guide_net = [nn.Conv2d(ci, channel, kernel, stride=1, padding=kernel//2), nn.ReLU()]
        for _ in range(layer - 2):
            if res:
                # 假设外部已定义 ResidualBlockNoBN，若无请保持原样或替换为标准Conv
                guide_net += [ResidualBlockNoBN(channel)]
            else:
                guide_net += [nn.Conv2d(channel, channel, kernel, stride=1, padding=kernel//2),
                              nn.ReLU()]
        guide_net += [nn.Conv2d(channel, co, kernel, stride=1, padding=kernel//2), nn.Tanh()]  # nn.Tanh()
        self.guide_net = nn.Sequential(*guide_net)

    def forward(self, x):
        return self.guide_net(x)


class Slicing(nn.Module):
    """
    向量化实现（优化版）：
    1. 移除 CPU->GPU 同步和 Python 循环。
    2. 直接在 GPU 上生成坐标网格，数学逻辑保持不变。
    """
    def __init__(self):
        super().__init__()

    def forward(self, grid, guide):
        # grid: [B_total, 3, D, Hg, Wg]
        # guide: [B_total, 1, H, W]

        B, _, H, W = guide.shape
        device = guide.device
        dtype = guide.dtype

        # --- 优化点：直接在 GPU 生成网格 ---

        # 生成 X 轴坐标 (W)
        xx = torch.arange(W, device=device, dtype=dtype)
        xx = xx.mul_(2.0 / W).sub_(1.0) # [W]

        # 生成 Y 轴坐标 (H)
        yy = torch.arange(H, device=device, dtype=dtype)
        yy = yy.mul_(2.0 / H).sub_(1.0) # [H]

        # 生成 Meshgrid [H, W] -> [B, H, W]
        grid_y, grid_x = torch.meshgrid(yy, xx, indexing='ij')

        # 扩展维度 [B, H, W]
        grid_x = grid_x.unsqueeze(0).expand(B, -1, -1)
        grid_y = grid_y.unsqueeze(0).expand(B, -1, -1)

        # 增加 Channel 维度并堆叠 [B, H, W, 3] (grid_sample 需要 channels last: x, y, z)
        # guide 是 z 轴坐标: [B, 1, H, W] -> permute -> [B, H, W, 1]
        guide_z = guide.permute(0, 2, 3, 1)

        # 组合: stack(x, y, z) -> [B, H, W, 3]
        sampling_grid = torch.stack([grid_x, grid_y, guide_z.squeeze(-1)], dim=-1)

        # 增加 depth 维度以匹配 grid_sample 的 5D 输入要求
        # Sampling grid: [B, 1, H, W, 3] (模拟 depth=1 的 3D 采样)
        sampling_grid = sampling_grid.unsqueeze(1)

        # 采样
        sampled = F.grid_sample(
            grid,
            sampling_grid,
            padding_mode='border',
            align_corners=False
        )
        # output: [B, 3, 1, H, W]

        return sampled.squeeze(2) # [B, 3, H, W]


class apply_coeff(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, coeff_split):
        # x: [B, 3, H, W]
        # coeff_split: [B, 3, 4, H, W] (前3通道是weight, 第4通道是bias)

        B, c, H, W = x.shape

        weights = coeff_split[:, :, :c, :, :] # [B, 3, 3, H, W]
        bias = coeff_split[:, :, c, :, :]     # [B, 3, H, W]

        # --- 优化点：使用 einsum 替代广播乘法 ---
        # 修正前的错误公式: 'boihw, bihMw -> bohw' (多了一个 M)
        # 修正后的正确公式: 'boihw, bihw -> bohw'
        # b:batch, o:out_c, i:in_c, h:height, w:width
        # 含义: 对 input channel (i) 进行求和
        out = torch.einsum('boihw, bihw -> bohw', weights, x)

        out = out + bias
        return out


class EnhBlock(nn.Module):
    def __init__(self, co, kernel=1, layer=2):
        super().__init__()
        self.context1 = GuideNN(ci=3, co=co, kernel=kernel, layer=layer, res=False, channel=16, scale=1)
        self.slicing = Slicing()
        self.apply_coeff = apply_coeff()
        self.activation = nn.ReLU() # 保留定义

    def forward(self, src, grid):
        # src: [B, 3, H_img, W_img]
        # grid: [B, 12, D, Hg, Wg]

        # 1. 计算 Context Map
        context_map = self.context1(src) # [B, 4, H_img, W_img]

        # 获取维度信息
        B, _, D, Hg, Wg = grid.shape
        B_ctx, C_ctx, H_img, W_img = context_map.shape

        # --- 优化点：Batch Merging (合并批处理) ---

        # 2. Grid 重塑
        # [B, 12, D, Hg, Wg] -> [B, 4, 3, D, Hg, Wg] -> [B*4, 3, D, Hg, Wg]
        grid_batch = grid.view(B, 4, 3, D, Hg, Wg).reshape(B * 4, 3, D, Hg, Wg)

        # 3. Context 重塑
        # [B, 4, H_img, W_img] -> [B, 4, 1, H_img, W_img] -> [B*4, 1, H_img, W_img]
        # 使用 H_img, W_img (448) 而不是 grid 的 Hg, Wg (56)
        context_batch = context_map.unsqueeze(2).reshape(B * 4, 1, H_img, W_img)

        # 4. 并行执行 Slicing
        # 输出: [B*4, 3, H_img, W_img]
        coeff_batch = self.slicing(grid_batch, context_batch)

        # 5. 还原维度
        # [B*4, 3, H_img, W_img] -> [B, 4, 3, H_img, W_img]
        coeff = coeff_batch.view(B, 4, 3, H_img, W_img)

        # 6. 调整顺序以适配 apply_coeff
        # [B, 4, 3, H, W] -> [B, 3, 4, H, W]
        coeff_split = coeff.permute(0, 2, 1, 3, 4)

        # 7. 应用系数
        out = self.apply_coeff(src, coeff_split)

        return out


class EnhancementNet(nn.Module):
    def __init__(self, ci=3,grid_range=8, scale_factor=0.5,co=1,kernel=1,layer=2,width = 16,enc_blks = [2, 2, 2, 2],dec_blks = [2, 2, 2, 2], middle_blk_num=2):
        super().__init__()
        self.relu = nn.ReLU()
        self.scale_factor = scale_factor
        self.mlp1 = EnhBlock(co=co, kernel=kernel, layer=layer)
        self.mlp2 = EnhBlock(co=co, kernel=kernel, layer=layer)
        self.mlp3 = EnhBlock(co=co, kernel=kernel, layer=layer)
        self.ci = ci
        self.grid_range = grid_range
        # 假设 EncDec 在外部定义，这里保持调用不变
        self.model = EncDec(input_channel=self.ci, width=width, middle_blk_num=middle_blk_num,
                    enc_blk_nums=enc_blks, dec_blk_nums=dec_blks,final_dim_1=12*grid_range,final_dim_2=12*grid_range,final_dim_3=12*grid_range)

    def forward(self, x):

        x_down = F.interpolate(x, scale_factor=self.scale_factor, mode='area')
        grid1,grid2,grid3 = self.model(x_down)

        b1, c1, h1, w1 = grid1.shape
        grid1 = grid1.reshape(b1,c1//self.grid_range,self.grid_range,h1,w1)

        b2, c2, h2, w2 = grid2.shape
        grid2 = grid2.reshape(b2,c2//self.grid_range,self.grid_range,h2,w2)

        b3, c3, h3, w3 = grid3.shape
        grid3 = grid3.reshape(b3,c3//self.grid_range,self.grid_range,h3,w3)

        res = x
        x = self.mlp1(x, grid1)
        x = self.relu(x)
        x = self.mlp2(x, grid2)
        x = self.relu(x)

        out = self.mlp3(x, grid3)
        return out + res


class DualStageISP(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.model_config = load_model_config(config)
        self.res = RestorationNet(**self.model_config['restoration'])
        self.enh = EnhancementNet(**self.model_config['enhancement'])

    def forward(self, raw: torch.Tensor):
        pro_predict = self.res(raw)
        out = self.enh(pro_predict)
        return out, pro_predict
