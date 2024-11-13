import numpy as np
import random
from torch.utils.data import Dataset
from feeders import tools

class SkeletonDataset(Dataset):
    def __init__(self, data_path, label_path=None, interval=1, split='train', data_format='j',
                 augment_type='z', intra_prob=0.5, inter_prob=0.0, win_size=-1,
                 debug_mode=False, threshold=64, uniform_sample=False, part_segment=False):

        self.debug_mode = debug_mode
        self.data_path = data_path
        self.label_path = label_path
        self.split = split
        self.data_format = data_format
        self.augment_type = augment_type
        self.intra_prob = intra_prob
        self.inter_prob = inter_prob
        self.win_size = win_size
        self.interval = interval
        self.threshold = threshold
        self.uniform_sample = uniform_sample
        self.part_segment = part_segment
        self._load_data()
        
        if part_segment:
            self._initialize_body_parts()

    def _initialize_body_parts(self):
        # Body part indices for partitioned data
        self.right_arm = np.array([7, 8, 22, 23]) - 1
        self.left_arm = np.array([11, 12, 24, 25]) - 1
        self.right_leg = np.array([13, 14, 15, 16]) - 1
        self.left_leg = np.array([17, 18, 19, 20]) - 1
        self.h_torso = np.array([5, 9, 6, 10]) - 1
        self.w_torso = np.array([2, 3, 1, 4]) - 1
        self.body_parts = np.concatenate((self.right_arm, self.left_arm, self.right_leg, 
                                          self.left_leg, self.h_torso, self.w_torso), axis=-1)

    def _load_data(self):
        data = np.load(self.data_path)
        
        if self.split == 'train':
            self.data = data['x_train']
            self.label = np.where(data['y_train'] > 0)[1]
            self.sample_names = [f'train_{i}' for i in range(len(self.data))]
        elif self.split == 'test':
            self.data = data['x_test']
            self.label = np.where(data['y_test'] > 0)[1]
            self.sample_names = [f'test_{i}' for i in range(len(self.data))]
        else:
            raise ValueError("Supported splits are 'train' and 'test' only")
            
        N, T, _ = self.data.shape
        self.data = self.data.reshape((N, T, 2, 25, 3)).transpose(0, 4, 1, 3, 2)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        sample_data = self.data[index]
        label = self.label[index]
        sample_data = np.array(sample_data)
        valid_frames = np.sum(sample_data.sum(0).sum(-1).sum(-1) != 0)
        people_count = np.sum(sample_data.sum(0).sum(0).sum(0) != 0)

        if self.uniform_sample:
            sample_data, frame_idx = tools.valid_crop_uniform(sample_data, valid_frames, 
                                                              self.interval, self.win_size, self.threshold)
        else:
            sample_data, frame_idx = tools.valid_crop_resize(sample_data, valid_frames, 
                                                             self.interval, self.win_size, self.threshold)

        if self.split == 'train':
            sample_data = self._apply_augmentations(sample_data, frame_idx, label, people_count)

        if self.data_format == 'b':
            sample_data = tools.joint2bone()(sample_data)
        elif self.data_format == 'jm':
            sample_data = tools.to_motion(sample_data)
        elif self.data_format == 'bm':
            sample_data = tools.joint2bone()(sample_data)
            sample_data = tools.to_motion(sample_data)

        if self.part_segment:
            sample_data = sample_data[:, :, self.body_parts]

        return sample_data, frame_idx, label, index

    def _apply_augmentations(self, data, frame_idx, label, people_count):
        p = np.random.rand()
        
        if p < self.intra_prob:
            if 'a' in self.augment_type and np.random.rand() < 0.5:
                data = data[:, :, :, np.array([1, 0])]

            if 'b' in self.augment_type and people_count == 2 and np.random.rand() < 0.5:
                axis_next = np.random.randint(0, 1)
                temp = data.copy()
                temp[:, :, :, axis_next] = np.zeros((data.shape[0], data.shape[1], data.shape[2]))
                data = temp

            if '1' in self.augment_type:
                data = tools.shear(data, p=0.5)
            if '2' in self.augment_type:
                data = tools.rotate(data, p=0.5)
            if '3' in self.augment_type:
                data = tools.scale(data, p=0.5)
            if '4' in self.augment_type:
                data = tools.spatial_flip(data, p=0.5)
            if '5' in self.augment_type:
                data, frame_idx = tools.temporal_flip(data, frame_idx, p=0.5)
            if '6' in self.augment_type:
                data = tools.gaussian_noise(data, p=0.5)
            if '7' in self.augment_type:
                data = tools.gaussian_filter(data, p=0.5)
            if '8' in self.augment_type:
                data = tools.drop_axis(data, p=0.5)
            if '9' in self.augment_type:
                data = tools.drop_joint(data, p=0.5)
                
        elif self.intra_prob <= p < (self.intra_prob + self.inter_prob):
            other_idx = random.choice(np.where(self.label == label)[0])
            alt_data = self.data[other_idx]
            f_num = np.sum(alt_data.sum(0).sum(-1).sum(-1) != 0)
            temporal_idx = np.round((frame_idx + 1) * f_num / 2).astype(int)
            data = tools.skeleton_adain_bone_length(data, alt_data[:, temporal_idx])
            
        return data.copy()

    def top_k(self, scores, top_k):
        ranked_scores = scores.argsort()
        top_hits = [lbl in ranked_scores[i, -top_k:] for i, lbl in enumerate(self.label)]
        return sum(top_hits) / len(top_hits)

def import_module(name):
    components = name.split('.')
    module = __import__(components[0])
    for comp in components[1:]:
        module = getattr(module, comp)
    return module
