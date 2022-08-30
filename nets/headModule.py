import torch
import torch.nn as nn
from nets.attention import ChannelAttention
from collections import OrderedDict
class headModule(nn.Module):
  def __init__(self, in_channel):
    super(headModule, self).__init__()
    
    spatial_kernelsize = 3
    base_channel = 64
    
    self.masterConv = nn.Sequential(
      self.conv2d(in_channel, base_channel, 3),
      self.conv2d(base_channel, 16, 1),
      self.conv2d(16, base_channel, 3)
    )
    self.depthConv  = nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1, groups=in_channel)
    self.channelAtt = ChannelAttention(in_channel + base_channel, 4)
    self.outConv = self.conv2d(in_channel + base_channel, base_channel, 1)
  
  def conv2d(self, in_channel, out_channel, kernel_size):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
      ("classLib_conv", nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, padding=pad, bias=False)),
      ("classLib_norm", nn.BatchNorm2d(out_channel)),
      ("classLib_relu", nn.LeakyReLU(0.1))])
      )
  def forward(self, x):
    
    x_master = self.masterConv(x)
    x_branch = self.depthConv(x)
    
    x_all = torch.cat([x_master, x_branch], 1)
    
    x_channelAtt = self.channelAtt(x_all)
    x_all = x_all * x_channelAtt
    out   = self.outConv(x_all)
    return out
