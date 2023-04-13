#coding:utf-8
import torch
import argparse
import torch.nn as nn
import torch.optim as opt
from PIL import Image
import argparse
import numpy as np
from albumentations import Compose, Resize, Normalize, ColorJitter, HorizontalFlip, VerticalFlip
import os

from learnerable_seg import PromptSAM
from scheduler import PolyLRScheduler

parser = argparse.ArgumentParser("Learnable prompt")
parser.add_argument("--image", type=str, required=True, 
                    help="path to the image that used to train the model")
parser.add_argument("--mask_path", type=str, required=True,
                    help="path to the mask file for training")
parser.add_argument("--epoch", type=int, default=32, 
                    help="training epochs")
parser.add_argument("--checkpoint", type=str, required=True,
                    help="path to the checkpoint of sam")
parser.add_argument("--model_name", default="default", type=str,
                    help="name of the sam model, default is vit_h",
                    choices=["default", "vit_b", "vit_l", "vit_h"])
parser.add_argument("--save_path", type=str, default="./ckpt_prompt",
                    help="save the weights of the model")
parser.add_argument("--num_classes", type=int, default=12)
parser.add_argument("--mix_precision", action="store_true", default=False,
                    help="whether use mix precison training")
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
parser.add_argument("--optimizer", default="adamw", type=str,
                    help="optimizer used to train the model")
parser.add_argument("--weight_decay", default=5e-4, type=float, 
                    help="weight decay for the optimizer")
parser.add_argument("--momentum", default=0.9, type=float,
                    help="momentum for the sgd")

def batch_inter_union(pred, target, num_classes):
    if pred.dim() == 4:
        pred = torch.max(pred, dim=1)[1]
    mini = 1
    maxi = num_classes
    nbins = num_classes
    pred = pred.long() + 1
    target = target.long() + 1
    pred = pred * (target > 0)
    target = target * (target > 0)
    intersection = pred * (pred == target)
    area_inter = torch.histc(intersection, bins=nbins, min=mini, max=maxi)
    area_out = torch.histc(pred, bins=nbins, min=mini, max=maxi)
    area_target = torch.histc(target, bins=nbins, min=mini, max=maxi)
    area_union = area_out + area_target - area_inter
    assert torch.sum(area_inter > area_union).item() == 0, "Intersection area should be smaller than Union area"
    return area_inter.cpu(), area_union.cpu()

def comput_iou(output, target, eps=1e-9):
    num_classes = output.size(1)
    inter, union = batch_inter_union(output, target, num_classes)
    iou = inter / (union + eps)
    return iou.mean()

def main(args):
    img_path = args.image
    mask_path = args.mask_path
    epochs = args.epoch
    checkpoint = args.checkpoint
    model_name = args.model_name
    save_path = args.save_path
    optimizer = args.optimizer
    weight_decay = args.weight_decay
    momentum = args.momentum
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    num_classes = args.num_classes
    model = PromptSAM(model_name, checkpoint=checkpoint, num_classes=num_classes)
    img = Image.open(img_path).convert("RGB")
    img = np.asarray(img)
    mask = Image.open(mask_path).convert("L")
    mask = np.asarray(mask)
    # pixel_mean=[123.675, 116.28, 103.53],
    # pixel_std=[58.395, 57.12, 57.375],
    # pixel_mean = np.array(pixel_mean) / 255
    # pixel_std = np.array(pixel_std) / 255
    pixel_mean = [0.5]*3
    pixel_std = [0.5]*3
    lr = args.lr
    transform = Compose(
        [
            ColorJitter(),
            VerticalFlip(),
            HorizontalFlip(),
            Resize(1024, 1024),
            Normalize(mean=pixel_mean, std=pixel_std)
        ]
    )
    scaler = torch.cuda.amp.grad_scaler.GradScaler()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    if optimizer == "adamw":
        optim = opt.AdamW([{"params":model.parameters(), "initia_lr": lr}], lr=lr, weight_decay=weight_decay)
    elif optimizer == "sgd":
        optim = opt.SGD([{"params":model.parameters(), "initia_lr": lr}], lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=True)
    loss_func = nn.CrossEntropyLoss()
    scheduler = PolyLRScheduler(optim, num_images=1, batch_size=1, epochs=epochs)
    best_iou = 0.
    for epoch in range(epochs):
        aug_data = transform(image=img, mask=mask)
        x = aug_data["image"]
        target = aug_data["mask"]
        x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(device)
        target = torch.from_numpy(target).unsqueeze(0).to(device, dtype=torch.long)
        optim.zero_grad()
        if device_type == "cuda" and args.mix_precision:
            x = x.to(dtype=torch.float16)
            with torch.autocast(device_type=device_type, dtype=torch.float16):
                pred = model(x)
                loss = loss_func(pred, target)

            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            x = x.to(detype=torch.float32)
            pred = model(x)
            loss = loss_func(pred, target)
            loss.backward()
            optim.step()
        iou = comput_iou(torch.softmax(pred, dim=1), target)
        scheduler.step()
        print("epoch-{}:{} iou:{}".format(epoch, loss, iou.item()))
        if iou > best_iou:
            best_iou = iou
            torch.save(
                model.state_dict(), os.path.join(save_path, "sam_{}_prompt.pth".format(model_name))
            )

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)