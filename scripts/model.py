#!/usr/bin/env python3
"""
NAFNet - Nonlinear Activation Free Network
State-of-the-art image restoration with excellent efficiency.
Paper: https://arxiv.org/abs/2204.04676

Optimized for real estate photo enhancement.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    """Channel-wise Layer Normalization."""
    
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, dim, 1, 1))
    
    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True) + 1e-6
        return self.weight * (x - mean) / std + self.bias


class SimpleGate(nn.Module):
    """Simple Gate - splits channels and multiplies (no activation!)."""
    
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    """
    Core NAFNet block.
    Key innovation: No nonlinear activations (ReLU, GELU, etc.)
    """
    
    def __init__(self, dim, ffn_expansion=2, dropout=0.0):
        super().__init__()
        
        hidden = dim * ffn_expansion
        
        # Spatial attention branch
        self.norm1 = LayerNorm(dim)
        
        # dim -> hidden*2 -> (gate) -> hidden -> hidden*2 -> (gate) -> hidden
        self.conv1 = nn.Conv2d(dim, hidden * 2, 1)
        self.gate1 = SimpleGate()  # hidden*2 -> hidden
        
        self.conv2 = nn.Conv2d(hidden, hidden * 2, 3, padding=1, groups=hidden)
        self.gate2 = SimpleGate()  # hidden*2 -> hidden
        
        # Simplified Channel Attention (preserves hidden channels)
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden, hidden, 1)  # FIXED: hidden -> hidden (not hidden -> dim)
        )
        
        # Project back to dim
        self.conv3 = nn.Conv2d(hidden, dim, 1)
        
        # FFN branch
        self.norm2 = LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, hidden * 2, 1),
            SimpleGate(),
            nn.Conv2d(hidden, dim, 1)
        )
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Learnable scaling factors
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
    
    def forward(self, x):
        # Spatial attention branch
        y = self.norm1(x)
        y = self.conv1(y)       # dim -> hidden*2
        y = self.gate1(y)       # hidden*2 -> hidden
        y = self.conv2(y)       # hidden -> hidden*2
        y = self.gate2(y)       # hidden*2 -> hidden
        y = y * self.sca(y)     # hidden * hidden -> hidden (element-wise)
        y = self.conv3(y)       # hidden -> dim
        y = self.dropout(y)
        x = x + y * self.beta
        
        # FFN branch
        y = self.norm2(x)
        y = self.ffn(y)
        y = self.dropout(y)
        x = x + y * self.gamma
        
        return x


class NAFNet(nn.Module):
    """
    NAFNet for image enhancement.
    
    Architecture:
    - Encoder-decoder with skip connections
    - NAFBlocks instead of convolutions
    - Global residual learning (output = input + learned_edit)
    
    Config for real estate photos (~17M params):
    - channels: [32, 64, 128, 256]
    - enc_blocks: [2, 2, 4, 4]
    - dec_blocks: [2, 2, 2]
    - middle_blocks: 4
    """
    
    def __init__(
        self,
        in_ch=3,
        out_ch=3,
        channels=[32, 64, 128, 256],
        enc_blocks=[2, 2, 4, 4],
        dec_blocks=[2, 2, 2],
        middle_blocks=4,
    ):
        super().__init__()
        
        # Initial feature extraction
        self.intro = nn.Conv2d(in_ch, channels[0], 3, padding=1)
        
        # Final output projection
        self.outro = nn.Conv2d(channels[0], out_ch, 3, padding=1)
        
        # Encoder
        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        
        for i, (ch, num_blocks) in enumerate(zip(channels, enc_blocks)):
            self.encoders.append(
                nn.Sequential(*[NAFBlock(ch) for _ in range(num_blocks)])
            )
            if i < len(channels) - 1:
                # Downsample: stride-2 convolution
                self.downs.append(
                    nn.Conv2d(ch, channels[i + 1], 2, stride=2)
                )
        
        # Middle (bottleneck)
        self.middle = nn.Sequential(
            *[NAFBlock(channels[-1]) for _ in range(middle_blocks)]
        )
        
        # Decoder
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        
        dec_channels = list(reversed(channels[:-1]))
        for i, (ch, num_blocks) in enumerate(zip(dec_channels, dec_blocks)):
            up_ch = channels[-(i + 1)]
            # Upsample: transposed convolution
            self.ups.append(
                nn.ConvTranspose2d(up_ch, ch, 2, stride=2)
            )
            self.decoders.append(
                nn.Sequential(*[NAFBlock(ch) for _ in range(num_blocks)])
            )
        
        # Learnable skip connection scaling
        self.skip_scale = nn.ParameterList([
            nn.Parameter(torch.ones(1)) for _ in dec_blocks
        ])
    
    def forward(self, x):
        inp = x  # Save for global residual
        
        x = self.intro(x)
        
        # Encoder path
        skips = []
        for i, encoder in enumerate(self.encoders):
            x = encoder(x)
            skips.append(x)
            if i < len(self.downs):
                x = self.downs[i](x)
        
        # Middle
        x = self.middle(x)
        
        # Decoder path
        skips = skips[:-1][::-1]  # Reverse, exclude last
        for i, (up, decoder) in enumerate(zip(self.ups, self.decoders)):
            x = up(x)
            # Handle size mismatch
            if x.shape != skips[i].shape:
                x = F.interpolate(x, size=skips[i].shape[2:], mode='bilinear', align_corners=True)
            # Skip connection with learnable scale
            x = x + skips[i] * self.skip_scale[i]
            x = decoder(x)
        
        x = self.outro(x)
        
        # Global residual: output = input + learned_edit
        return x + inp


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_info():
    """Get model information."""
    model = NAFNet()
    params = count_parameters(model)
    
    return {
        "name": "NAFNet",
        "parameters": f"{params:,}",
        "input_shape": "(B, 3, 1024, 1024)",
        "output_shape": "(B, 3, 1024, 1024)",
        "channels": [32, 64, 128, 256],
        "enc_blocks": [2, 2, 4, 4],
        "dec_blocks": [2, 2, 2],
    }


if __name__ == "__main__":
    print("=" * 50)
    print("NAFNet Model Test")
    print("=" * 50)
    
    # Test model creation
    model = NAFNet()
    params = count_parameters(model)
    print(f"\nParameters: {params:,}")
    
    # Test forward pass with small input first
    print("\nTesting forward pass...")
    
    for size in [64, 256, 512]:
        x = torch.randn(1, 3, size, size)
        try:
            with torch.no_grad():
                y = model(x)
            print(f"  {size}x{size}: {x.shape} -> {y.shape} ✓")
        except Exception as e:
            print(f"  {size}x{size}: FAILED - {e}")
    
    print("\n✓ Model test complete!")