# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    @author: Zhongxi Qiu
    @create time: 2021/4/13 10:49
    @filename: confusion_matrix.py
    @software: PyCharm
"""
import torch

def confusion_matrix(output, target, num_classes):
    """
    Calculate confusion matrix
    Args:
        output: output of the model, [bs, num_classes, w,h]
        target: true label, [bs, w,h]
        num_classes: number of classes

    Returns:
        the confusion matrix
    """
    y_pred = output.clone().detach().flatten()
    y = target.clone().detach().flatten()

    target_mask = (y >= 0) & (y < num_classes)
    y = y[target_mask]
    y_pred = y_pred[target_mask]
    indices = num_classes * y + y_pred
    matrix = torch.bincount(indices, minlength=num_classes**2).reshape(num_classes, num_classes)
    return matrix

def confusion_matrix_v2(y_pred, y_true, num_classes):
    N = num_classes
    y_true = y_true.clone().detach().flatten().long()
    y_pred = y_pred.clone().detach().flatten().long()
    return torch.sparse.LongTensor(
        torch.stack([y_true, y_pred]),
        torch.ones_like(y_true, dtype=torch.long),
        torch.Size([N, N])).to_dense()