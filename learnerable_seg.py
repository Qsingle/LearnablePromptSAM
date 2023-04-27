#coding:utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F

from segment_anything.modeling.common import LayerNorm2d
from segment_anything.modeling.image_encoder import Block
from segment_anything import sam_model_registry
from dino_vit import vit_base, vit_small, vit_large, vit_giant2

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

class FFTPrompt(nn.Module):
    def __init__(self, rate=0.25, prompt_type="highpass") -> None:
        super(FFTPrompt, self).__init__()
        assert prompt_type in ["highpass", "lowpass"], "The prompt type must in " \
        "['highpass', 'lowpass'], but got {}".format(prompt_type)
        self.rate = rate
        self.prompt_type = prompt_type
    
    def forward(self, x):
        fft = torch.fft.fft2(x, norm="forward")
        fft = torch.fft.fftshift(fft)
        h, w = x.shape[2:]
        radio = int((h*w*self.rate)**.5 // 2)
        mask = torch.zeros_like(x)
        c_h, c_w = h // 2, w // 2
        mask[:, :, c_h-radio:c_h+radio, c_w-radio:c_w+radio] = 0
        if self.prompt_type == "highpass":
            fft = fft*(1-mask)
        else:
            fft = fft * mask
        real, imag = fft.real, fft.imag
        shift = torch.fft.fftshift(torch.complex(real, imag))
        inv = torch.fft.ifft2(shift, norm="forward")
        inv = inv.real
        return torch.abs(inv)

class PromptGen(nn.Module):
    def __init__(self, blk, reduction=4, cls_token=False, reshape=False, seq_size=None) -> None:
        super(PromptGen, self).__init__()
        self.block = blk
        dim = blk.attn.qkv.in_features
        prompt_dim = dim // reduction
        self.prompt_learn = nn.Sequential(
            # nn.Linear(dim, 32),
            # nn.GELU(),
            # nn.Linear(32, dim),
            # nn.GELU()
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
        self.cls_token = cls_token
        self.reshape = reshape
        self.seq_size = seq_size
    
    def forward(self, x):
        if self.cls_token:
            tokens = x[:,1:]
            bs, seq_len, dim = tokens.size()
            if self.reshape:
                tokens = tokens.reshape(-1, self.seq_size, self.seq_size, dim).permute(0, 3, 1, 2)
            prompt = self.prompt_learn(tokens)
            promped = tokens + prompt
            promped = promped.reshape(bs, dim, seq_len).transpose(1, 2)
            promped = torch.cat([x[:, 0].unsqueeze(1), promped], dim=1)
        else:
            prompt = self.prompt_learn(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            # prompt = self.prompt_learn(x)
            promped = x + prompt
        net = self.block(promped)
        return net

class PromptSAM(nn.Module):
    def __init__(self, model_name, checkpoint, num_classes=12, reduction=4, upsample_times=2, groups=4, 
                 prompt_input=False, prompt_type="fft", fft_type="highpass", freq_num=0.25) -> None:
        super(PromptSAM, self).__init__()
        #load same from the pretrained model
        self.sam = sam_model_registry[model_name](checkpoint=checkpoint)
        del self.sam.prompt_encoder
        del self.sam.mask_decoder
        out_dim = self.sam.image_encoder.neck[0].out_channels
        for param in self.sam.image_encoder.parameters():
            param.requires_grad = False
        self.img_size = self.sam.image_encoder.img_size
        blocks = []
        for block in self.sam.image_encoder.blocks:
            blocks.append(
                PromptGen(block, reduction=reduction)
            )
        self.sam.image_encoder.blocks = nn.Sequential(
            *blocks
        )
        self.up_conv = nn.ModuleDict()
        self.up_times = upsample_times
        dim = out_dim
        for i in range(upsample_times):
            self.up_conv["up_{}".format(i+1)] = nn.Sequential(
                    # nn.Conv2d(dim, dim // 2, 1, 1, 0),
                    nn.ConvTranspose2d(dim, dim//2, 2, 2),
                    LayerNorm2d(dim // 2),
                    nn.GELU()
                )
            dim = dim // 2
        self.ms_conv = MSConv2d(dim, groups=groups)
        self.decoder = nn.Sequential(
            nn.Conv2d(dim, num_classes, 1, 1, 0),
        )
        
        if prompt_input:
            if prompt_type == "fft":
                self.prompt_input = FFTPrompt(rate=freq_num, prompt_type=fft_type)
        else:
            self.prompt_input = nn.Identity()

    def upscale(self, x, times=2):
        for i in range(times):
            # x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
            x = self.up_conv["up_{}".format(i+1)](x)
        return x

    def forward(self, x):
        out = self.sam.image_encoder(x)
        out = self.upscale(out, self.up_times)
        out = self.ms_conv(out)
        seg_out = self.decoder(out)
        seg_out = F.interpolate(seg_out, size=(self.img_size, self.img_size), mode="bilinear", align_corners=True)
        return seg_out

DINO_VIT_RESITRY = {
    "vit_b": vit_base,
    "vit_s": vit_small,
    "vit_l": vit_large,
    "vit_g": vit_giant2
}

DINO_CFG = {
    "vit_s":  {
              "patch_size": 14,
              "drop_path_rate": 0.4,
              "ffn_layer": "mlp",
              "block_chunks": 0,
              "img_size": 518,
              "init_values": 1e-5
        
            },
    "vit_l":  {
              "patch_size": 14,
              "drop_path_rate": 0.4,
              "ffn_layer": "mlp",
              "block_chunks": 0,
              "img_size": 518,
              "init_values": 1e-5
        
            },
    "vit_b": {
              "patch_size": 14,
              "drop_path_rate": 0.4,
              "ffn_layer": "mlp",
              "block_chunks": 0,
              "img_size": 518,
              "init_values": 1e-5
    },
    "vit_g": {
              "patch_size": 14,
              "drop_path_rate": 0.4,
              "ffn_layer": "swiglufused",
              "block_chunks": 0,
              "img_size": 518,
              "init_values": 1e-5
    },
}

class PromptDiNo(nn.Module):
    def __init__(self, name, checkpoint=None, reduction=4, num_classes=12, upsample_times=2, groups=4) -> None:
        super().__init__()
        cfg = DINO_CFG[name]
        self.encoder = DINO_VIT_RESITRY[name](**cfg)
        self.reset_backbone(checkpoint)
        for param in self.encoder.parameters():
            param.requires_grad = False
        dim = self.encoder.norm.normalized_shape[0]
        blks = []
        self.patch_size = cfg["patch_size"]
        self.img_size = cfg['img_size']
        self.fea_size = self.img_size // self.patch_size
        for blk in self.encoder.blocks:
             blks.append(PromptGen(blk, reduction=reduction, cls_token=True, reshape=True, seq_size=self.fea_size))
        self.encoder.blocks = nn.Sequential(*blks)
        out_dim = self.encoder.num_features
        dim = out_dim
        self.upscale_times = upsample_times
        self.up_conv = nn.ModuleDict()
        for i in range(upsample_times):
            self.up_conv["up_{}".format(i+1)] = nn.Sequential(
                    # nn.Conv2d(dim, dim // 2, 1, 1, 0),
                    nn.ConvTranspose2d(dim, dim//2, 2, 2),
                    LayerNorm2d(dim // 2),
                    nn.GELU()
                )
            dim = dim // 2
        self.ms_conv = MSConv2d(dim, groups=groups)
        self.out_conv = nn.Conv2d(dim, num_classes, 1, 1, 0) 

    
    def reset_backbone(self, chekpoint=None):
        if chekpoint is None:
            return
        state = torch.load(chekpoint, map_location="cpu")
        self.encoder.load_state_dict(state)
    
    def upscale(self, x):
        for i in range(self.upscale_times):
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
            x = self.up_conv["up_{}".format(i+1)](x)
        return x

    
    def forward(self, x):
        featrues = self.encoder.forward_features(x)
        feature = featrues["x_norm_patchtokens"]
        bs, _, dim = feature.shape
        feature = feature.reshape(bs, self.img_size // self.patch_size, self.img_size // self.patch_size, dim).permute(0, 3, 1, 2)
        feature = self.upscale(feature)
        feature = self.ms_conv(feature)
        out = self.out_conv(feature) 
        out = torch.nn.functional.interpolate(out, size=(self.img_size, self.img_size), mode="bilinear", align_corners=True)
        return out


if __name__ == "__main__":
    with torch.no_grad():
        # model = PromptSAM("vit_b", "ckpts/sam_vit_b_01ec64.pth").half().cuda()
        x = torch.randn(1, 3, 518, 518).half().cuda()
        cfg = {
              "patch_size": 14,
              "drop_path_rate": 0.4,
            #   "ffn_layer": "swiglufused",
              "block_chunks": 0,
              "img_size": 512,
              "init_values": 1e-5
        
        }
        model = PromptDiNo("vit_s", "ckpts/dinov2_vits14_pretrain.pth", 4).half().cuda()

        out = model(x)
        print(out.shape)