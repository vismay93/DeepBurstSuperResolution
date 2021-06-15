import torch
import torch.nn as nn 
from utils.warp import warp
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

def get_tensor_grid(tensor_shape):
    tenHor = torch.linspace(-1.0 + (1.0 / tensor_shape[3]), 1.0 - (1.0 / tensor_shape[3]), tensor_shape[3]).view(1, 1, 1, -1).expand(-1, -1, tensor_shape[2], -1)
    tenVer = torch.linspace(-1.0 + (1.0 / tensor_shape[2]), 1.0 - (1.0 / tensor_shape[2]), tensor_shape[2]).view(1, 1, -1, 1).expand(-1, -1, -1, tensor_shape[3])
    return torch.cat([ tenHor, tenVer ], 1).cuda()


class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, conv_shortcut=False, dropout=0.2):
        super(ResnetBlock, self).__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = torch.nn.GroupNorm(num_groups=16, num_channels=in_channels, eps=1e-6, affine=True)
        #self.norm1 = torch.nn.BatchNorm2d(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        self.norm2 = torch.nn.GroupNorm(num_groups=16, num_channels=out_channels, eps=1e-6, affine=True)
        self.dropout = torch.nn.Dropout(dropout)
        #self.norm2 = torch.nn.BatchNorm2d(out_channels)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)
        
        self.nonlinearity = nn.ReLU()

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = self.nonlinearity(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = self.nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h


class Encoder(nn.Module):
    def __init__(self, in_ch=3, hidden_ch=64, out_ch=512, num_blocks=9):
        super(Encoder, self).__init__()
        self.conv_in = nn.Conv2d(in_ch, hidden_ch, 3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(hidden_ch, out_ch, 3, stride=1, padding=1)
        self.nonlinearity = nn.ReLU()
        self.res_blocks = nn.Sequential(*[ResnetBlock(hidden_ch, hidden_ch) for _ in range(num_blocks)])

    def forward(self, x):
        x = self.conv_in(x)
        x = self.nonlinearity(x)
        x = self.res_blocks(x)
        x = self.conv_out(x)
        x = self.nonlinearity(x)
        return x


class Aligner(nn.Module):
    def __init__(self, in_ch=512, hidden_ch=64):
        super(Aligner, self).__init__()
        self.conv_in = nn.Conv2d(in_ch, hidden_ch, 3, stride=1, padding=1)
        self.nonlinearity = nn.ReLU()
        
    def forward(self, feat, flow):
        feat = self.conv_in(feat)
        feat = self.nonlinearity(feat)
        feat = warp(feat, flow)
        return feat


class Fusion(nn.Module):
    def __init__(self, in_ch=512, align_ch=64, hidden_ch=128, out_ch=512, num_blocks=3):
        super(Fusion, self).__init__()
        self.conv_flow = nn.Conv2d(2, align_ch, 3, stride=1, padding=1)
        self.res_block_flow = ResnetBlock(align_ch, align_ch)
        self.conv_in = nn.Conv2d(3*align_ch, hidden_ch, 3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(hidden_ch, out_ch, 3, stride=1, padding=1)
        self.nonlinearity = nn.ReLU()
        self.res_blocks = nn.Sequential(*[ResnetBlock(hidden_ch, hidden_ch) for _ in range(num_blocks)])

    def forward(self, base, feat, flow_mod1):
        flow_mod1 = self.conv_flow(flow_mod1)
        flow_mod1 = self.nonlinearity(flow_mod1)
        x = torch.cat([feat, feat-base, flow_mod1], axis=1)
        x = self.conv_in(x)
        x = self.nonlinearity(x)
        x = self.res_blocks(x)
        x = self.conv_out(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_ch=512, hidden_ch=64, out_ch=3, num_blocks=[5,4], scale=4):
        super(Decoder, self).__init__()
        upscale_factor = 2*scale
        self.conv_in = nn.Conv2d(in_ch, hidden_ch, 3, stride=1, padding=1)
        self.conv_sr = nn.Conv2d(hidden_ch, hidden_ch*upscale_factor**2, 3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(hidden_ch, out_ch, 3, stride=1, padding=1)
        self.nonlinearity = nn.ReLU()
        self.res_blocks1 = nn.Sequential(*[ResnetBlock(hidden_ch, hidden_ch) for _ in range(num_blocks[0])])
        self.res_blocks2 = nn.Sequential(*[ResnetBlock(hidden_ch, hidden_ch) for _ in range(num_blocks[1])])
        self.pixel_shuffle = Rearrange('b (c h2 w2) h w -> b c (h h2) (w w2)', h2=upscale_factor, w2=upscale_factor)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.conv_in(x)
        x = self.nonlinearity(x)
        x = self.res_blocks1(x)
        x = self.conv_sr(x)
        x = self.pixel_shuffle(x)
        x = self.res_blocks2(x)
        x = self.conv_out(x)
        x = self.sigmoid(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self):
        super(ChannelAttention, self).__init__()
        self.nonlinearity = nn.Sigmoid()
    
    def forward(self, w):
        w = self.nonlinearity(w)
        w_m = reduce(w, 'b k c h w -> b c h w', 'sum')
        w_m = repeat(w_m, 'b c h w -> b k c h w', k=w.shape[1])
        w = w / w_m 
        return w