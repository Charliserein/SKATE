import random
import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

def crop_resize(data, valid_frames, interval, window, min_frames):
    C, T, V, M = data.shape
    start = 0
    end = valid_frames
    total_frames = end - start

    # Determine cropping size
    if len(interval) == 1:
        crop_ratio = interval[0]
        crop_offset = int((1 - crop_ratio) * total_frames / 2)
        cropped_data = data[:, start + crop_offset:end - crop_offset, :, :]
        crop_start, crop_end = start + crop_offset, end - crop_offset
    else:
        crop_ratio = np.random.uniform(interval[0], interval[1])
        cropped_length = np.clip(int(total_frames * crop_ratio), min_frames, total_frames)
        crop_offset = np.random.randint(0, total_frames - cropped_length + 1)
        cropped_data = data[:, start + crop_offset:start + crop_offset + cropped_length, :, :]
        crop_start, crop_end = start + crop_offset, start + crop_offset + cropped_length

    # Resize
    cropped_data = torch.tensor(cropped_data, dtype=torch.float).permute(2, 3, 0, 1).contiguous()
    reshaped_data = cropped_data.view(V * M, C, cropped_data.shape[-1])
    resized_data = F.interpolate(reshaped_data, size=window, mode='linear', align_corners=False)
    output_data = resized_data.view(V, M, C, window).permute(2, 3, 0, 1).numpy()

    time_indices = torch.arange(start=crop_start, end=crop_end, dtype=torch.float)
    scaled_time_indices = F.interpolate(time_indices[None, None, :], size=window, mode='linear', align_corners=False)
    return output_data, (2 * scaled_time_indices / total_frames - 1).numpy().squeeze()

def uniform_crop_resize(data, valid_frames, interval, window, min_frames):
    C, T, V, M = data.shape
    start, end = 0, valid_frames
    total_frames = end - start

    # Determine cropping parameters
    if len(interval) == 1:
        crop_ratio = interval[0]
        cropped_length = np.clip(int(total_frames * crop_ratio), min_frames, total_frames)
        crop_offset = int((1 - crop_ratio) * total_frames / 2)
        indices = np.arange(cropped_length) + crop_offset
        cropped_data = data[:, indices, :, :]
    else:
        crop_ratio = np.random.uniform(interval[0], interval[1])
        cropped_length = np.clip(int(total_frames * crop_ratio), min_frames, total_frames)
        crop_offset = np.random.randint(0, total_frames - cropped_length + 1)

        if cropped_length < window:
            indices = np.arange(cropped_length)
        else:
            step = cropped_length // window
            indices = np.arange(0, cropped_length, step)[:window] + crop_offset

        cropped_data = data[:, indices, :, :]

    # Resize
    cropped_data = torch.tensor(cropped_data, dtype=torch.float).permute(2, 3, 0, 1).contiguous()
    reshaped_data = cropped_data.view(V * M, C, len(indices))
    if len(indices) != window:
        resized_data = F.interpolate(reshaped_data, size=window, mode='linear', align_corners=False)
        indices_tensor = F.interpolate(torch.tensor(indices, dtype=torch.float)[None, None, :], size=window,
                                       mode='linear', align_corners=False).squeeze()
    else:
        resized_data, indices_tensor = reshaped_data, indices

    output_data = resized_data.view(V, M, C, window).permute(2, 3, 0, 1).numpy()
    return output_data, (2 * indices_tensor / total_frames - 1).numpy()

def add_scale(data, scale_factor=0.2, prob=0.5):
    if random.random() < prob:
        scale = 1 + np.random.uniform(-1, 1, (3, 1, 1, 1)) * np.array(scale_factor)
        return data * scale
    return data.copy()

def subtract_reference(data, prob=0.5):
    if random.random() < prob:
        joint_idx = random.randint(0, data.shape[2] - 1)
        ref_data = data[:, :, joint_idx, :]
        return data - ref_data[:, :, None, :]
    return data.copy()

def random_flip(data, indices, prob=0.5):
    if random.random() < prob:
        time_range = list(range(data.shape[1]))
        flipped_data = data[:, time_range[::-1], :, :]
        return flipped_data, -indices
    return data.copy(), indices.copy()

def flip_spatial(data, prob=0.5):
    if random.random() < prob:
        transform_idx = {'uav': [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]}
        return data[:, :, transform_idx['uav'], :]
    return data.copy()

def apply_rotation(data, axis=None, angle=None, prob=0.5):
    if random.random() < prob:
        axis = axis or random.randint(0, 2)
        angle = math.radians(angle or random.uniform(-30, 30))
        cos_angle, sin_angle = math.cos(angle), math.sin(angle)
        rot_matrix = {
            0: np.array([[1, 0, 0], [0, cos_angle, sin_angle], [0, -sin_angle, cos_angle]]),
            1: np.array([[cos_angle, 0, -sin_angle], [0, 1, 0], [sin_angle, 0, cos_angle]]),
            2: np.array([[cos_angle, sin_angle, 0], [-sin_angle, cos_angle, 0], [0, 0, 1]])
        }[axis]
        return np.dot(data.transpose(1, 2, 3, 0), rot_matrix).transpose(3, 0, 1, 2)
    return data.copy()

class GaussianBlur(nn.Module):
    def __init__(self, channels=3, kernel_size=15, sigma_range=(0.1, 2), prob=0.5):
        super(GaussianBlur, self).__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.sigma_range = sigma_range
        self.prob = prob
        self.kernel = None
        self.create_kernel()

    def create_kernel(self):
        sigma = random.uniform(*self.sigma_range)
        radius = self.kernel_size // 2
        kernel = np.exp(-np.square(np.arange(-radius, radius + 1)) / (2 * sigma**2))
        kernel = torch.from_numpy(kernel).view(1, 1, -1).repeat(self.channels, 1, 1, 1)
        self.kernel = nn.Parameter(kernel, requires_grad=False)

    def forward(self, x):
        if random.random() < self.prob:
            x = torch.from_numpy(x).double().permute(3, 0, 2, 1)
            x = F.conv2d(x, self.kernel, padding=(0, self.kernel_size // 2), groups=self.channels)
            return x.permute(1, -1, -2, 0).numpy()
        return x.numpy()

''' Skeleton AdaIN '''
def skeleton_adain_bone_length(input, ref): # C T V M
    eps = 1e-5
    center = 1
    ref_c = ref[:, :, center, :]

    # joint to bone (joint2bone)
    j2b = joint2bone()
    bone_i = j2b(input) # C T V M
    bone_r = j2b(ref)

    bone_length_i = np.linalg.norm(bone_i, axis=0) # T V M
    bone_length_r = np.linalg.norm(bone_r, axis=0)

    bone_length_scale = (bone_length_r + eps) / (bone_length_i + eps) # T V M
    bone_length_scale = np.expand_dims(bone_length_scale, axis=0) # 1 T V M

    bone_i = bone_i * bone_length_scale

    # bone to joint (bone2joint)
    b2j = bone2joint()
    joint = b2j(bone_i, ref_c)
    return joint

#换成我们的图节点（<16)
class joint2bone(nn.Module):
    def __init__(self):
        super(joint2bone, self).__init__()
        self.pairs = [(10, 8), (8, 6), (9, 7), (7, 5), (15, 13), (13, 11), (16, 14), (14, 12), (11, 5), (12, 6), (11, 12), (5, 6), (5, 0), (6, 0), (1, 0), (2, 0), (3, 1), (4, 2)]
        
    def __call__(self, joint):
        bone = np.zeros_like(joint)
        for v1, v2 in self.pairs:
            bone[:, :, v1, :] = joint[:, :, v1, :] - joint[:, :, v2, :]
        return bone

#改了图节点
class bone2joint(nn.Module):
    def __init__(self):
        super(bone2joint, self).__init__()
        self.center = 0
        self.pairs_1 = [(10, 8)]
        self.pairs_2 = [(8, 6), (9, 7), (7, 5)]
        self.pairs_3 = [(15, 13), (13, 11), (16, 14), (14, 12)]
        self.pairs_4 = [(11, 5), (12, 6), (11, 12)]
        self.pairs_5 = [(5, 6), (5, 0), (6, 0)]
        self.pairs_6 = [(1, 0), (2, 0), (3, 1), (4, 2)]

    def __call__(self, bone, center):
        joint = np.zeros_like(bone)
        joint[:, :, self.center, :] = center
        for v1, v2 in self.pairs_1:
            joint[:, :, v1, :] = bone[:, :, v1, :] + joint[:, :, v2, :]
        for v1, v2 in self.pairs_2:
            joint[:, :, v1, :] = bone[:, :, v1, :] + joint[:, :, v2, :]
        for v1, v2 in self.pairs_3:
            joint[:, :, v1, :] = bone[:, :, v1, :] + joint[:, :, v2, :]
        for v1, v2 in self.pairs_4:
            joint[:, :, v1, :] = bone[:, :, v1, :] + joint[:, :, v2, :]
        for v1, v2 in self.pairs_5:
            joint[:, :, v1, :] = bone[:, :, v1, :] + joint[:, :, v2, :]
        for v1, v2 in self.pairs_6:
            joint[:, :, v1, :] = bone[:, :, v1, :] + joint[:, :, v2, :]
        return joint

def to_motion(input): # C T V M
    C, T, V, M = input.shape
    motion = np.zeros_like(input)
    motion[:, :T - 1] = np.diff(input, axis=1)
    return motion
