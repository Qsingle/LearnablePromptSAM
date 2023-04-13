#coding:utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F

from segment_anything.modeling.common import LayerNorm2d
from segment_anything import sam_model_registry

class ImagePool(nn.Module):
    def __init__(self, in_ch):
        super(ImagePool, self).__init__()
        self.gpool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(in_ch, in_ch, 1, 1)

    def forward(self, x):
        net = self.gpool(x)
        net = self.conv(net)
        net = F.interpolate(net, size=x.size()[2:], mode="bilinear", align_corners=False)
        return net

class MSConv2d(nn.Module):
    def __init__(self, ch, groups=4):
        super(MSConv2d, self).__init__()
        assert ch % groups == 0
        group_ch = ch // groups
        self.convs = nn.ModuleList([
            nn.Conv2d(group_ch, group_ch, 1, 1)
        ])
        for i in range(1, groups):
            self.convs.append(
                nn.Conv2d(group_ch, group_ch, 3, 1, padding=i, dilation=i, groups=group_ch)
            )
        # self.convs.append(ImagePool(group_ch))
        self.activate = nn.GELU()
        self.norm = nn.BatchNorm2d(ch)
        self.groups = groups

    def forward(self, x):
        features = x.chunk(self.groups, dim=1)
        outs = []
        for i in range(len(features)):
            outs.append(self.convs[i](features[i]))
        net = torch.cat(outs, dim=1)
        net = self.norm(net)
        net = self.activate(net)
        return net

class PromptGen(nn.Module):
    def __init__(self, blk, reduction=4) -> None:
        super(PromptGen, self).__init__()
        self.block = blk
        dim = blk.attn.qkv.in_features
        prompt_dim = dim // reduction
        self.prompt_learn = nn.Sequential(
            nn.Conv2d(dim, prompt_dim, 1, 1),
            LayerNorm2d(prompt_dim),
            nn.GELU(),
            nn.Conv2d(prompt_dim, prompt_dim, 3, 1, 1, groups=prompt_dim, bias=False),
            LayerNorm2d(prompt_dim),
            nn.GELU(),
            nn.Conv2d(prompt_dim, dim, 1, 1),
            LayerNorm2d(dim),
            nn.GELU()
        )
    
    def forward(self, x):
        prompt = self.prompt_learn(x)
        promped = x + prompt
        net = self.block(promped)
        return net

class PromptSAM(nn.Module):
    def __init__(self, model_name, checkpoint, num_classes=12, reduction=4) -> None:
        super(PromptSAM, self).__init__()
        #load same from the pretrained model
        self.sam = sam_model_registry[model_name](checkpoint=checkpoint)
        del self.sam.prompt_encoder
        del self.sam.mask_decoder
        out_dim = self.sam.image_encoder.neck[0].out_channels
        for param in self.sam.image_encoder.parameters():
            param.requires_grad = False
        self.img_size = self.sam.image_encoder.img_size
        for block in self.sam.image_encoder.blocks:
            block = PromptGen(block, reduction=reduction)
        self.up_conv = nn.ModuleDict(
            {
                "up_1": nn.Sequential(
                    nn.Conv2d(out_dim, out_dim // 4, 1, 1, 0),
                    LayerNorm2d(out_dim // 4),
                    nn.GELU()
                ),
                "up_2": nn.Sequential(
                    nn.Conv2d(out_dim // 4, out_dim // 8, 1, 1, 0),
                    LayerNorm2d(out_dim // 8),
                    nn.GELU()
                )
            }
        )
        self.ms_conv = MSConv2d(out_dim//8, groups=4)
        self.decoder = nn.Sequential(
            nn.Conv2d(out_dim // 8, num_classes, 1, 1, 0),
        )

    def upscale(self, x, times=2):
        for i in range(times):
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
            x = self.up_conv["up_{}".format(i+1)](x)
        return x

    def forward(self, x):
        out = self.sam.image_encoder(x)
        out = self.upscale(out)
        out = self.ms_conv(out)
        seg_out = self.decoder(out)
        seg_out = F.interpolate(seg_out, size=(self.img_size, self.img_size), mode="bilinear", align_corners=True)
        return seg_out


if __name__ == "__main__":
    with torch.no_grad():
        model = PromptSAM("vit_b", "ckpts/sam_vit_b_01ec64.pth").half().cuda()
        x = torch.randn(1, 3, 1024, 1024).half().cuda()
        out = model(x)
        print(out.shape)