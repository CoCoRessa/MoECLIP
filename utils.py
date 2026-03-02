import os
import random
import numpy as np
import torch
from torch.nn import functional as F
import kornia as K
from torchvision import transforms


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def get_rot_mat(theta):
    theta = torch.tensor(theta)
    return torch.tensor(
        [
            [torch.cos(theta), -torch.sin(theta), 0],
            [torch.sin(theta), torch.cos(theta), 0],
        ]
    )


def get_translation_mat(a, b):
    return torch.tensor([[1, 0, a], [0, 1, b]])


def rot_img(x, scale):
    theta = scale
    dtype = torch.FloatTensor
    if x.dim() == 3:
        x = x.unsqueeze(0)
    rot_mat = get_rot_mat(theta)[None, ...].type(dtype).repeat(x.shape[0], 1, 1)
    grid = F.affine_grid(rot_mat, x.size()).type(dtype)
    x = F.grid_sample(x, grid, padding_mode="reflection")
    x = x.squeeze(0)
    return x


def translation_img(x, translation):
    a, b = translation
    dtype = torch.FloatTensor
    if x.dim() == 3:
        x = x.unsqueeze(0)
    rot_mat = get_translation_mat(a, b)[None, ...].type(dtype).repeat(x.shape[0], 1, 1)
    grid = F.affine_grid(rot_mat, x.size()).type(dtype)
    x = F.grid_sample(x, grid, padding_mode="reflection")
    x = x.squeeze(0)
    return x


def hflip_img(x, **kwargs):
    if x.dim() == 3:
        x = x.unsqueeze(0)
    x = K.geometry.transform.hflip(x)
    x = x.squeeze(0)
    return x


def vflip_img(x, **kwargs):
    if x.dim() == 3:
        x = x.unsqueeze(0)
    x = K.geometry.transform.vflip(x)
    x = x.squeeze(0)
    return x


def add_gaussian_noise(x, scale=0.05):
    std = scale
    noise_mask = torch.randn(x.shape[-2:]) > 3
    noise = torch.randn_like(x) * std  # mean = 0
    noised_img = (x + noise) * noise_mask
    noise_img = torch.where(noised_img > 0, noised_img, x)
    return noise_img


def cos_sim(a_norm, b_norm):
    if len(a_norm.shape) == 2:
        sim_mt = b_norm @ a_norm.transpose(1, 0)
    elif len(a_norm.shape) == 1:
        sim_mt = b_norm @ a_norm
    else:
        raise NotImplementedError
    return sim_mt

class AddGaussianNoise(object):
    def __init__(self, std=1.0, p=0.5):
        self.std = std
        self.p = p

    def __call__(self, x):

        if random.random() < self.p:
            return x
        if not isinstance(x, torch.Tensor):
            x = transforms.ToTensor()(x)
        noise_mask = (torch.randn(x.shape[-2:]) > 3).int()
        noise = torch.randn_like(x) * self.std  # mean = 0
        noised_img = (1 - noise_mask) * x + noise * x * noise_mask
        noised_img = torch.clamp(noised_img, 0.0, 1.0)
        return transforms.ToPILImage()(noised_img)

    def __repr__(self):
        return self.__class__.__name__ + f"(std={self.std}, p={self.p})"
