import torch
import torch.nn.functional as F
import numpy as np

def label_to_onehot(target, num_classes=10):
    target = torch.unsqueeze(target, 1)
    onehot_target = torch.zeros(target.size(0), num_classes, device=target.device)
    onehot_target.scatter_(1, target, 1)
    return onehot_target


def cross_entropy_for_onehot(pred, target):
    return torch.mean(torch.sum(-target * F.log_softmax(pred, dim=-1), 1))


def total_variation(x):
    """Anisotropic TV."""
    dx = torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))
    dy = torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
    return dx + dy

# 新增代码

def random_sparsify(image, sparsity):
    # mask = (torch.rand_like(image[0]) > sparsity).unsqueeze(0).repeat(3, 1, 1)
    mask = (torch.rand_like(image[0]) > sparsity)
    # print("image:",image.shape)
    # print("mask:",mask.shape)
    return mask

def random_sparsify_channels(image, sparsity):
    mask = (torch.rand_like(image) > sparsity).float()
    return image * mask

def cosine_similarity(vector1, vector2):
    dot_product = np.dot(vector1, vector2)
    norm_vector1 = np.linalg.norm(vector1)
    norm_vector2 = np.linalg.norm(vector2)
    cosine_similarity = dot_product / (norm_vector1 * norm_vector2)
    return cosine_similarity
