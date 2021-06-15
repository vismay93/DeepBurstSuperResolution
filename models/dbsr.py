import torch
import torch.nn as nn
from einops import rearrange, repeat, reduce

from .modules import Encoder, Aligner, Fusion, Decoder, ChannelAttention


class DeepBurstSR(nn.Module):
    def __init__(self, config=None):
        super(DeepBurstSR, self).__init__()
        self.config  = config 
        self.encode = Encoder(in_ch=4, hidden_ch=64, out_ch=256)
        self.align = Aligner(in_ch=256, hidden_ch=64)
        self.fuse = Fusion(in_ch=64, hidden_ch=128, out_ch=256)
        self.decode = Decoder(in_ch=256, out_ch=3, scale=4)
        self.softmax = ChannelAttention()
        
    def forward(self, x, flow):
        batch_size = x.shape[0]
        burst_size = x.shape[1]

        x = rearrange(x, 'b k c h w -> (b k) c h w')
        flow = rearrange(flow, 'b k c h w -> (b k) c h w')
        x = self.encode(x)
        x_aligned = self.align(x, flow)
        temp = rearrange(x_aligned, '(b k) c h w -> b k c h w', k=burst_size)
        x_base = repeat(temp[:,0,:,:,:], 'b c h w -> b k c h w', k=burst_size)
        x_base = rearrange(x_base, 'b k c h w -> (b k) c h w')
        w = self.fuse(x_base, x_aligned, torch.remainder(flow, 1))

        w = rearrange(w, '(b k) c h w -> b k c h w', k=burst_size)
        x = rearrange(x, '(b k) c h w -> b k c h w', k=burst_size)
        w = self.softmax(w)
        x = x * w
        x = reduce(x, 'b k c h w -> b c h w', 'sum')
        x = self.decode(x)
        return x
