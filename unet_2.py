from typing import Optional
import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class TimeEmbedding(nn.Module):
    """ 
    Embedding for 't', timestep 
    time_channels - Used to determine the MLP layer for the timestep embeddings
    """

    def __init__(self, time_channels):
      super().__init__()
      
      self.time_channels = time_channels
      # Define the MLP for the time embeddings
      self.time_mlp = nn.Sequential(
                                    nn.Linear(self.time_channels//4, self.time_channels),
                                    nn.SiLU(),
                                    nn.Linear(self.time_channels, self.time_channels)
                                    )
      
    
    def forward(self, t: torch.Tensor):
      """
      t - Similar to d_model or vector size
      half_dim - Max length for each position
      """

      # The max length is chosen as half_dim here
      half_dim = self.time_channels//8
      emb = (math.log(10000.0)/(half_dim-1))
      emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
      emb = t[:, None] * emb[None, :]
      emb = torch.cat((emb.sin(), emb.cos()), dim= 1)

      # Transform with MLP
      emb = self.time_mlp(emb)
      return emb


class ResidualBlock(nn.Module):
    """
    A residual block has two convolution layers with group normalization. Each resolution is processed by two
    convolutional blocks.
    """

    def __init__(self, in_channels: int, out_channels: int, time_channels: int, 
                n_groups: int = 32, dropout: float = 0.1):
      super().__init__()

      # Group normalization and first convolutional layer
      self.first_layer = nn.Sequential(
                                      nn.GroupNorm(n_groups, in_channels),
                                      nn.SiLU(),
                                      nn.Conv2d(in_channels, out_channels, 3, 1, 1)
                                      )
      # Group normalization and second convolutional layer
      self.second_layer = nn.Sequential(
                                        nn.GroupNorm(n_groups, out_channels),
                                        nn.SiLU(),
                                        nn.Dropout(dropout),
                                        nn.Conv2d(out_channels, out_channels, 3, 1, 1)
                                        )
      # If the number of in_channels is not equal to the number of out_channels we have to project 
      # the shortcut connection.
      if in_channels != out_channels:
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
      else:
        self.shortcut = nn.Identity()
      # Linear Layer for time embeddings
      self.time_emb = nn.Sequential(
                                    nn.Linear(time_channels, out_channels),
                                    nn.SiLU()
                                    )
    
    def forward(self, x: torch.Tensor, t: torch.Tensor):
      assert x.device == t.device

      h = self.first_layer(x)
      h += self.time_emb(t)[:, :, None, None]
      h = self.second_layer(h)
      h += self.shortcut(x)

      return h

class AttentionBlock(nn.Module):
    """
    Similar to transformer multi-head attention
    n_channels : Number of channels in the input
    n_heads : Number of heads in the multi-head attention
    d_k : Number of dimensions in each head
    n_groups : Number of groups for group normalization
    """

    def __init__(self, n_channels: int, n_heads: int = 1, d_k: int = None, n_groups: int = 32):
      super().__init__()

      # default value of d_k
      if d_k == None:
        d_k = n_channels
      # Sequential layers
      self.norm = nn.GroupNorm(n_groups, n_channels)
      self.projection = nn.Linear(n_channels, n_heads*d_k*3)
      self.output = nn.Linear(n_heads*d_k, n_channels)

      self.scale = d_k**-0.5
      self.n_heads = n_heads
      self.d_k = d_k

    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None):
      """
      x has shape : [batch_size, in_channels, height, width]
      t has shape : [batch_size, time_channels]
      """
      # Not used but kept as a signature to match with ResidualBlock
      # assert x.device == t.device
      _ = t
      # Get shape
      batch_size, n_channels, height, width = x.shape
      # Change x to shape [batch_size, seq, n_channels]
      x = x.view(batch_size, n_channels, -1).permute(0,2,1)
      # Get queries, key and values (concatenated) and shape it to [batch_size, seq, n_heads, 3*d_k]
      qkv = self.projection(x).view(batch_size, -1, self.n_heads, 3*self.d_k)
      # Split queries, key and values. Each of them will have shape [batch_size, seq, n_heads, d_k]
      q, k, v = torch.chunk(qkv, 3, dim=-1)
      # Calculate the scaled dot-product
      atten = torch.einsum('bihd, bjhd->bijh', q, k)*self.scale
      # Softmax along sequence direction
      atten = atten.softmax(dim=2)
      # Multiply by values
      res = torch.einsum('bijh, bjhd->bihd', atten, v)
      # Reshape to [batch_size, seq, n_heads * d_k]
      res = res.view(batch_size, -1, self.n_heads * self.d_k)
      # Transform to [batch_size, seq, n_channels]
      res = self.output(res)
      # Add skip connection
      res += x
      # Change to [batch_size, in_channels, height, width]
      res = res.permute(0, 2, 1).view(batch_size, n_channels, height, width)
      
      return res

      
class DownBlock(nn.Module):
    """
    Combines the Residual Block and the Attention Block, used in the first half of the UNet at each resolution.
    """

    def __init__(self, in_channels: int, out_channels: int, time_channels: int, has_atten: bool):
      super().__init__()

      self.res = ResidualBlock(in_channels, out_channels, time_channels)

      if has_atten:
          self.atten = AttentionBlock(out_channels)
      else:
          self.atten = nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor):
      
      t = t.to(x.device)
      assert x.device == t.device
      x = self.res(x, t)
      x = self.atten(x)

      return x

class UpBlock(nn.Module):
  """
  Combines the Residual Block and the Attention Block, used in the second half of the UNet at each resolution.
  """

  def __init__(self, in_channels: int, out_channels: int, time_channels: int, has_atten: bool):
    super().__init__()

    self.res = ResidualBlock(in_channels + out_channels, out_channels, time_channels)

    if has_atten:
      self.atten = AttentionBlock(out_channels)
    else:
      self.atten = nn.Identity()
    
  def forward(self, x: torch.Tensor, t: torch.Tensor):
    t = t.to(x.device)
    assert x.device == t.device
    x = self.res(x, t)
    x = self.atten(x)

    return x

class MiddleBlock(nn.Module):
  """
  Combines a Residual Block followed by a Attention Block and finally another Residual Block. 
  Applied at the lowest resolution of the UNet.
  """

  def __init__(self, n_channels: int, time_channels: int):
    super().__init__()

    self.res1 = ResidualBlock(n_channels, n_channels, time_channels)
    self.atten = AttentionBlock(n_channels)
    self.res2 = ResidualBlock(n_channels, n_channels, time_channels)

  def forward(self, x: torch.Tensor, t: torch.Tensor):
    t = t.to(x.device)
    assert x.device == t.device
    x = self.res1(x, t)
    x = self.atten(x)
    x = self.res2(x ,t)

    return x

class Upsample(nn.Module):
  """
  Scale the feature map up by 2 times.
  """

  def __init__(self, n_channels: int, ):
    super().__init__()

    self.conv = nn.ConvTranspose2d(n_channels, n_channels, 4, 2, 1)

  def forward(self, x: torch.Tensor, t: torch.Tensor):
    t = t.to(x.device)
    assert x.device == t.device
    _ = t
    return self.conv(x)

class Downsample(nn.Module):
  """
  Scale the feature map down by 2 times.
  """

  def __init__(self, n_channels: int):
    super().__init__()

    self.conv = nn.Conv2d(n_channels, n_channels, 3, 2, 1)

  def forward(self, x: torch.Tensor, t: torch.Tensor):
    t = t.to(x.device)
    assert x.device == t.device
    _ = t
    return self.conv(x)

class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.
    image_channels: Number of image channels. Eg 3 for RGB
    n_channels: Number of channels in the feature map that we transform the image into
    ch_mults: List of channels at each resolution. The number of channels is ch_mults[i]*n_channels
    is_atten: List of Booleans indicating whether to use attention at each resolution
    n_blocks: Number of UpDownBlocks at each resolution (Refers to the number of times the conv_filters are
    applied at each resolution step while upsampling and downsampling)

    Resolution here refers to the spatial dimension of the image as it goes through the UNet, usually the
    resolution is halved during downsampling and doubled while upsampling.
    """

    def __init__(self, image_channels = 3, n_channels = 32, ch_mults = (1, 2, 3, 4), is_atten = (False, False, True, True), n_blocks = 2, device = "cpu"):
        super().__init__()

        # Total number of Resolutions
        n_resolutions = len(ch_mults)

        # Project Image into Feature Map
        self.image_proj = nn.Conv2d(in_channels=image_channels, out_channels=n_channels, kernel_size=(3,3),
                                    padding=(1,1))
        
        # Time Embedding Layer - It has a total of (n_channels*4) channels
        time_channels = n_channels*4
        self.time_emb = TimeEmbedding(time_channels)

        """ First Half of UNet decreasing resolution """

        down = []
        out_channels, in_channels = n_channels, n_channels

        for i in range(n_resolutions):
          # Number of output channels at this resolution
          out_channels = in_channels*ch_mults[i]
          for j in range(n_blocks):
            # print("Channels: ", in_channels, out_channels)
            down.append(DownBlock(in_channels, out_channels, time_channels, is_atten[i]))
            in_channels = out_channels

          # Downsample at all resolutions except the last one
          if i < (n_resolutions-1):
            down.append(Downsample(in_channels))
          
        # Combines the set of modules
        self.down = nn.ModuleList(down)

        # Middle Block
        self.middle = MiddleBlock(out_channels, time_channels)
        """Second Half of UNet with increasing resolution"""

        up = []

        for i in reversed(range(n_resolutions)):
          # Number of output channels at this resolution
          out_channels = in_channels
          for j in range(n_blocks):
            up.append(UpBlock(in_channels, out_channels, time_channels, is_atten[i]))
            in_channels = out_channels

          out_channels = in_channels//ch_mults[i]
          up.append(UpBlock(in_channels, out_channels, time_channels, is_atten[i]))
          in_channels = out_channels

          # Upsample at all instances other than the first one
          if i > 0:
            up.append(Upsample(in_channels))

        # Combines the set of modules
        self.up = nn.ModuleList(up)

        # Final Normalization and convolution layer
        self.out = nn.Sequential(nn.GroupNorm(8, n_channels),
                                 nn.SiLU(),
                                 nn.Conv2d(in_channels, image_channels, kernel_size=3, padding=1)
                                )
        self.device = device

    def forward(self, x, t):
        """
          x - torch.tensor of the shape (batch_size, in_channels, height, width)
          t - torch.tensor of the shape (batch_size)
        """
        # Get timestep embeddings
        t_emb = self.time_emb(t).to(self.device)
        
        assert x.device == t.device

        # Get Image Projection
        x_proj = self.image_proj(x)

        # h will store outputs at each resolution for skip connection
        h = [x_proj]

        # First half of UNet - downsampling
        for m in self.down:
          x_proj = m(x_proj, t_emb)
          h.append(x_proj)

        assert x_proj.shape[0] == x.shape[0], x_proj.shape[1] == 512

        # Middle Block
        x = self.middle(x_proj, t_emb)

        # print("Here")

        # Second half of UNet - upsampling
        for n in self.up:
          if isinstance(n, Upsample):
            x = n(x, t_emb)
          else:
            # Get the skip connection from the first half of the UNet and concatenate
            s = h.pop()
            x = torch.cat((x,s), dim=1)
            x = n(x,t_emb)

        # Final output - Normalization, SiLU and convolution
        out = self.out(x)
        return out