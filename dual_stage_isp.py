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

                guide_net += [ResidualBlockNoBN(channel)]
            else:
                guide_net += [nn.Conv2d(channel, channel, kernel, stride=1, padding=kernel//2),
                              nn.ReLU()]
        guide_net += [nn.Conv2d(channel, co, kernel, stride=1, padding=kernel//2), nn.Tanh()]
        self.guide_net = nn.Sequential(*guide_net)

    def forward(self, x):
        return self.guide_net(x)

class Slicing(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, grid, guide):

        B, _, H, W = guide.shape
        device = guide.device
        dtype = guide.dtype

        xx = torch.arange(W, device=device, dtype=dtype)
        xx = xx.mul_(2.0 / W).sub_(1.0)

        yy = torch.arange(H, device=device, dtype=dtype)
        yy = yy.mul_(2.0 / H).sub_(1.0)

        grid_y, grid_x = torch.meshgrid(yy, xx, indexing='ij')

        grid_x = grid_x.unsqueeze(0).expand(B, -1, -1)
        grid_y = grid_y.unsqueeze(0).expand(B, -1, -1)

        guide_z = guide.permute(0, 2, 3, 1)

        sampling_grid = torch.stack([grid_x, grid_y, guide_z.squeeze(-1)], dim=-1)

        sampling_grid = sampling_grid.unsqueeze(1)

        sampled = F.grid_sample(
            grid,
            sampling_grid,
            padding_mode='border',
            align_corners=False
        )

        return sampled.squeeze(2)

class apply_coeff(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, coeff_split):

        B, c, H, W = x.shape

        weights = coeff_split[:, :, :c, :, :]
        bias = coeff_split[:, :, c, :, :]

        out = torch.einsum('boihw, bihw -> bohw', weights, x)

        out = out + bias
        return out

class EnhBlock(nn.Module):
    def __init__(self, co, kernel=1, layer=2):
        super().__init__()
        self.context1 = GuideNN(ci=3, co=co, kernel=kernel, layer=layer, res=False, channel=16, scale=1)
        self.slicing = Slicing()
        self.apply_coeff = apply_coeff()
        self.activation = nn.ReLU()

    def forward(self, src, grid):

        context_map = self.context1(src)

        B, _, D, Hg, Wg = grid.shape
        B_ctx, C_ctx, H_img, W_img = context_map.shape

        grid_batch = grid.view(B, 4, 3, D, Hg, Wg).reshape(B * 4, 3, D, Hg, Wg)

        context_batch = context_map.unsqueeze(2).reshape(B * 4, 1, H_img, W_img)

        coeff_batch = self.slicing(grid_batch, context_batch)

        coeff = coeff_batch.view(B, 4, 3, H_img, W_img)

        coeff_split = coeff.permute(0, 2, 1, 3, 4)

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
