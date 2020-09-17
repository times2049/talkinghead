import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as spectral_norm
import functools
import torch.nn.functional as F
from torch.nn import Parameter as P

class SNConv2d(nn.Module):
    def __init__(self, 
                in_channels: int, 
                out_channels: int, 
                kernel_size=3, 
                stride = 1, 
                padding = 0, 
                dilation = 1, 
                groups: int = 1, 
                bias: bool = True, 
                padding_mode: str = 'zeros',
                num_itrs = 1, 
                eps = 1e-12
            ):
        super(SNConv2d, self).__init__()
        self.snconv = functools.partial(
                                    spectral_norm,
                                    nn.Conv2d(
                                        in_channels, 
                                        out_channels, 
                                        kernel_size, 
                                        stride, 
                                        padding, 
                                        dilation, 
                                        groups, 
                                        bias
                                         ), 

                                    n_power_iterations = num_itrs, 
                                    eps = eps
                            ) 

    def forward(self, x):
        return self.snconv(x)

class BlockDown(nn.Module):
    def __init__(self, 
                    in_channels, 
                    out_channels, 
                    which_conv = SNConv2d, 
                    preactivation=True, 
                    activation=nn.ReLU(inplace=False), 
                    downsample=nn.AvgPool2d(2),
                    channel_ratio=4
                ):
        super(BlockDown, self).__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.hidden_channels = self.out_channels // channel_ratio
        self.which_conv = which_conv
        self.preactivation = preactivation
        self.activation = activation
        self.downsample = downsample
            
        # Conv layers
        self.conv1 = self.which_conv(self.in_channels, 
                                    self.hidden_channels, 
                                     kernel_size=1, padding=0)
        self.conv2 = self.which_conv(self.hidden_channels, self.hidden_channels)
        self.conv3 = self.which_conv(self.hidden_channels, self.hidden_channels)
        self.conv4 = self.which_conv(self.hidden_channels, self.out_channels, 
                                     kernel_size=1, padding=0)
                                     
        self.learnable_shortcut = True if (in_channels != out_channels) else False
        if self.learnable_shortcut:
            self.conv_shortcut = self.which_conv(in_channels, out_channels - in_channels, 
                                         kernel_size=1, padding=0)
    def shortcut(self, x):
        if self.downsample:
            x = self.downsample(x)
        if self.learnable_shortcut:
            x = torch.cat([x, self.conv_shortcut(x)], 1)    
        return x
    
    def forward(self, x):
        # 1x1 bottleneck conv
        h = self.conv1(F.relu(x))
        # 3x3 convs
        h = self.conv2(self.activation(h))
        h = self.conv3(self.activation(h))
        # relu before downsample
        h = self.activation(h)
        # downsample
        if self.downsample:
            h = self.downsample(h)     
        # final 1x1 conv
        h = self.conv4(h)
        return h + self.shortcut(x)

class Attention(nn.Module):
    def __init__(self, ch, which_conv=SNConv2d, name='attention'):
        super(Attention, self).__init__()
        # Channel multiplier
        self.ch = ch
        self.which_conv = which_conv
        self.theta = self.which_conv(self.ch, self.ch // 8, kernel_size=1, padding=0, bias=False)
        self.phi = self.which_conv(self.ch, self.ch // 8, kernel_size=1, padding=0, bias=False)
        self.g = self.which_conv(self.ch, self.ch // 2, kernel_size=1, padding=0, bias=False)
        self.o = self.which_conv(self.ch // 2, self.ch, kernel_size=1, padding=0, bias=False)
        # Learnable gain parameter
        self.gamma = P(torch.tensor(0.), requires_grad=True)
    def forward(self, x, y=None):
        # Apply convs
        theta = self.theta(x)
        phi = F.max_pool2d(self.phi(x), [2,2])
        g = F.max_pool2d(self.g(x), [2,2])    
        # Perform reshapes
        theta = theta.view(-1, self. ch // 8, x.shape[2] * x.shape[3])
        phi = phi.view(-1, self. ch // 8, x.shape[2] * x.shape[3] // 4)
        g = g.view(-1, self. ch // 2, x.shape[2] * x.shape[3] // 4)
        # Matmul and softmax to get attention maps
        beta = F.softmax(torch.bmm(theta.transpose(1, 2), phi), -1)
        # Attention map times g path
        o = self.o(torch.bmm(g, beta.transpose(1,2)).view(-1, self.ch // 2, x.shape[2], x.shape[3]))
        return self.gamma * o + x

        
class AdaptiveInstanceNorm(nn.Module):
    def __init__(self, in_channel, style_dim):
        super().__init__()

        self.norm = nn.InstanceNorm2d(in_channel)
        self.style = nn.Linear(style_dim, in_channel * 2)

        self.style.bias.data[:in_channel] = 1
        self.style.bias.data[in_channel:] = 0

    def forward(self, input, style):
        style = self.style(style).unsqueeze(2).unsqueeze(3)
        gamma, beta = style.chunk(2, 1)

        out = self.norm(input)
        out = gamma * out + beta

        return out   

class Block(nn.Module):
  def __init__(self, in_channels, out_channels, style_dim=512,
               which_conv=SNConv2d, which_in=AdaptiveInstanceNorm, activation=nn.ReLU(inplace=False),
               upsample=functools.partial(F.interpolate, scale_factor=2)
                                    , channel_ratio=4):
    super(Block, self).__init__()
    
    self.in_channels, self.out_channels = in_channels, out_channels
    self.hidden_channels = self.in_channels // channel_ratio
    self.which_conv, self.which_in = which_conv, which_in
    self.activation = activation
    # Conv layers
    self.conv1 = self.which_conv(self.in_channels, self.hidden_channels, 
                                 kernel_size=1, padding=0)
    self.conv2 = self.which_conv(self.hidden_channels, self.hidden_channels)
    self.conv3 = self.which_conv(self.hidden_channels, self.hidden_channels)
    self.conv4 = self.which_conv(self.hidden_channels, self.out_channels, 
                                 kernel_size=1, padding=0)
    if which_in:
        # Batchnorm layers
        self.in1 = self.which_in(self.in_channels, style_dim)
        self.in2 = self.which_in(self.hidden_channels, style_dim)
        self.in3 = self.which_in(self.hidden_channels, style_dim)
        self.in4 = self.which_in(self.hidden_channels, style_dim)
    # upsample layers
    self.upsample = upsample

  def forward(self, x, y):
    # Project down to channel ratio
    if which_in:
        h= self.in1(h, y)
    h = self.conv1(self.activation(h))
    # Apply next BN-ReLU
    if which_in:
        h = self.in2(h, y)
    h = self.activation(h)
    # Drop channels in x if necessary
    if self.in_channels != self.out_channels:
      x = x[:, :self.out_channels]      
    # Upsample both h and x at this point  
    if self.upsample:
      h = self.upsample(h)
      x = self.upsample(x)
    # 3x3 convs
    h = self.conv2(h)
    if which_in:
        h = self.in3(h, y)
    h = self.conv3(self.activation(h))
    # Final 1x1 conv
    if which_in:
        h = self.in4(h, y)
    h = self.conv4(self.activation(h))
    return h + x

    