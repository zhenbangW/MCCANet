import torch
import torch.nn as nn
from collections import OrderedDict
class ccaModule(nn.Module):
  def __init__(self, in_channel, class_num):
    super(ccaModule, self).__init__()
    
    self.class_num = class_num
    
    self.coarseConv = nn.Sequential(
      self.conv2d(in_channel, in_channel // 2, 1),
      self.conv2d(in_channel // 2, in_channel, 3),
      self.conv2d(in_channel, class_num, 1)
    )
    
    self.branchConv = nn.Sequential(
      self.conv2d(in_channel, in_channel // 2, 1),
      self.conv2d(in_channel // 2, in_channel, 3)
    )
    self.w          = nn.Sequential(
      self.conv2d(in_channel, in_channel//2, 1),
      self.conv2d(in_channel//2, in_channel, 3)
    )
  def conv2d(self, in_channel, out_channel, kernel_size):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
      ("classLib_conv", nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, padding=pad, bias=False)),
      ("classLib_norm", nn.BatchNorm2d(out_channel)),
      ("classLib_relu", nn.LeakyReLU(0.1))])
      )
  def forward(self, x):
    batch_size = x.size(0)
    in_h       = x.size(2)
    in_w       = x.size(3)
    
    
    coarseFeat = self.coarseConv(x)
    
    coarseFeat1 = coarseFeat.view(batch_size, -1, in_h * in_w).permute(0,2,1).contiguous()
    
    branchFeat = self.branchConv(x)
    branchFeat = branchFeat.view(batch_size, -1, in_h * in_w)
    
    
    cfl        = torch.matmul(branchFeat, coarseFeat1)
    cfl        = torch.nn.functional.softmax(cfl, dim=1)
    
    
    coarseFeat2 = coarseFeat.view(batch_size, -1, in_h * in_w)
    
    
    ccAtten        = torch.matmul(cfl, coarseFeat2).view(batch_size, -1, in_h, in_w)
    
    ccAtten        = self.w(ccAtten)
    
    return ccAtten, coarseFeat
