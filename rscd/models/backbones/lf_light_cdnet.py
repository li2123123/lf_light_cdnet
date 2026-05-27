import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class _ConvBNAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )


class _FrequencyEnhancement(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden_channels = max(channels // reduction, 8)
        self.low_proj = _ConvBNAct(channels, channels, kernel_size=1, padding=0)
        self.high_proj = _ConvBNAct(channels, channels, kernel_size=3, padding=1)
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.out_proj = _ConvBNAct(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        low = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        high = x - low
        low = self.low_proj(low)
        high = self.high_proj(high)
        gate = self.gate(torch.cat([low, high], dim=1))
        return self.out_proj(gate * low + (1 - gate) * high + x)


class _ConvLiquidCell(nn.Module):
    def __init__(self, channels, min_tau=0.1):
        super().__init__()
        self.min_tau = min_tau
        self.tau = nn.Conv2d(channels * 3, channels, kernel_size=3, padding=1)
        self.gate = nn.Conv2d(channels * 3, channels, kernel_size=3, padding=1)
        self.candidate = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Tanh(),
        )
        self.out_proj = _ConvBNAct(channels, channels, kernel_size=3, padding=1)

    def forward(self, x1, diff, x2):
        h = x1
        for x in (diff, x2):
            inputs = torch.cat([x, h, diff], dim=1)
            tau = F.softplus(self.tau(inputs)) + self.min_tau
            gate = torch.sigmoid(self.gate(inputs))
            candidate = self.candidate(inputs)
            h = h + 0.5 / tau * (-h + gate * candidate)
        return self.out_proj(h)


class _FrequencyLiquidFusion(nn.Module):
    def __init__(self, channels, use_liquid=False):
        super().__init__()
        self.use_liquid = use_liquid
        self.diff_proj = _ConvBNAct(channels * 3, channels, kernel_size=3, padding=1)
        self.liquid = _ConvLiquidCell(channels) if use_liquid else None
        fusion_channels = channels * 2 if use_liquid else channels
        self.fusion = _ConvBNAct(fusion_channels, channels, kernel_size=3, padding=1)

    def forward(self, f1, f2):
        diff = self.diff_proj(torch.cat([torch.abs(f2 - f1), f1, f2], dim=1))
        if not self.use_liquid:
            return diff
        liquid = self.liquid(f1, diff, f2)
        return self.fusion(torch.cat([diff, liquid], dim=1))


class _LightFPNDecoder(nn.Module):
    def __init__(self, in_channels, decoder_channels=96, num_classes=2, align_corners=False):
        super().__init__()
        self.align_corners = align_corners
        self.lateral = nn.ModuleList([
            _ConvBNAct(ch, decoder_channels, kernel_size=1, padding=0) for ch in in_channels
        ])
        self.smooth = nn.ModuleList([
            _ConvBNAct(decoder_channels, decoder_channels, kernel_size=3, padding=1) for _ in in_channels
        ])
        self.classifier = nn.Sequential(
            _ConvBNAct(decoder_channels, decoder_channels, kernel_size=3, padding=1),
            nn.Dropout2d(0.1),
            nn.Conv2d(decoder_channels, num_classes, kernel_size=1),
        )

    def forward(self, features, output_size):
        laterals = [layer(x) for layer, x in zip(self.lateral, features)]
        x = laterals[-1]
        x = self.smooth[-1](x)
        for idx in range(len(laterals) - 2, -1, -1):
            x = F.interpolate(x, size=laterals[idx].shape[2:], mode='bilinear', align_corners=self.align_corners)
            x = self.smooth[idx](x + laterals[idx])
        out = self.classifier(x)
        return F.interpolate(out, size=output_size, mode='bilinear', align_corners=self.align_corners)


class LFLightCDNet(nn.Module):
    def __init__(
        self,
        pretrained=True,
        channels=(64, 128, 256, 512),
        decoder_channels=96,
        num_classes=2,
        liquid_stages=(2, 3),
        align_corners=False,
        **kwargs,
    ):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        encoder = models.resnet18(weights=weights)
        self.stem = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu, encoder.maxpool)
        self.layer1 = encoder.layer1
        self.layer2 = encoder.layer2
        self.layer3 = encoder.layer3
        self.layer4 = encoder.layer4

        self.freq = nn.ModuleList([_FrequencyEnhancement(ch) for ch in channels])
        liquid_stages = set(liquid_stages)
        self.change_blocks = nn.ModuleList([
            _FrequencyLiquidFusion(ch, use_liquid=idx in liquid_stages) for idx, ch in enumerate(channels)
        ])
        self.decoder = _LightFPNDecoder(channels, decoder_channels, num_classes, align_corners)

    def _encode(self, x):
        x = self.stem(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x1, x2, x3, x4

    def forward(self, x1, x2):
        output_size = x1.shape[2:]
        f1 = self._encode(x1)
        f2 = self._encode(x2)
        change_features = []
        for idx, (a, b) in enumerate(zip(f1, f2)):
            a = self.freq[idx](a)
            b = self.freq[idx](b)
            change_features.append(self.change_blocks[idx](a, b))
        return self.decoder(change_features, output_size)
