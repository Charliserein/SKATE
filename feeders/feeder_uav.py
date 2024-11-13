import numpy as np
import random
from torch.utils.data import Dataset
from feeders import tools

class SkeletonDataset(Dataset):
    def __init__(self, data_path, label_path=None, interval=1, split='train', data_format='j',
                 aug_type='z', intra_prob=0.5, inter_prob=0.0, win_size=-1,
                 debug=False, threshold=64, uniform_crop=False, partition=False):
        """
        :param data_path: Path to the data file
        :param label_path: Path to the label file
        :param interval: Sampling interval for data segments
        :param split: Dataset split ('train' or 'test')
        :param data_format: Type of data to process (e.g., 'j' for joints)
        :param aug_type: Augmentation methods
        :param intra_prob: Probability for intra-instance augmentation
        :param inter_prob: Probability for inter-instance augmentation
        :param win_size: Window size for cropping
        :param debug: Debug mode flag
        :param threshold: Frame threshold
        :param uniform_crop: Flag for uniform cropping
        :param partition: Flag for partitioned processing
        """
        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.split = split
        self.data_format = data_format
        self.aug_type = aug_type
        self.intra_prob = intra_prob
        self.inter_prob = inter_prob
        self.win_size = win_size
        self.interval = interval
        self.threshold = threshold
        self.uniform_crop = uniform_crop
        self.partition = partition
        self._load_data()  # Load data and labels

    def _load_data(self):
        """Load data based on dataset split (train/test)"""
        self.data = np.load(self.data_path)
        self.label = np.load(self.label_path)
        self.sample_names = [f"{self.split}_{i}" for i in range(len(self.data))]

    def __len__(self):
        """Return dataset length"""
        return len(self.label)

    def __getitem__(self, index):
        """Retrieve and process a sample at the specified index"""
        data_sample = self.data[index]
        label = self.label[index]
        data_sample = np.array(data_sample)
        valid_frames = np.sum(data_sample.sum(axis=(0, -1, -1)) != 0)
        num_people = np.sum(data_sample.sum(axis=(0, 0, 0)) != 0)

        if valid_frames == 0:
            return np.zeros((3, self.win_size, 17, 2)), np.zeros((self.win_size,)), label, index

        # Crop data
        crop_fn = tools.valid_crop_uniform if self.uniform_crop else tools.valid_crop_resize
        data_sample, frame_indices = crop_fn(data_sample, valid_frames, self.interval, self.win_size, self.threshold)

        # Apply augmentation if in training mode
        if self.split == 'train':
            self._apply_augmentation(data_sample, frame_indices, num_people)

        # Convert to specified data type
        data_sample = self._convert_data_type(data_sample)

        return data_sample, frame_indices, label, index

    def _apply_augmentation(self, data_sample, frame_indices, num_people):
        """Apply intra- or inter-instance augmentation"""
        prob = np.random.rand()
        if prob < self.intra_prob:
            self._intra_instance_augmentation(data_sample, num_people)
        elif prob < (self.intra_prob + self.inter_prob):
            self._inter_instance_augmentation(data_sample, frame_indices)

    def _intra_instance_augmentation(self, data_sample, num_people):
        """Perform intra-instance augmentation based on aug_type"""
        aug_methods = {
            'a': lambda: data_sample[:, :, :, [1, 0]] if np.random.rand() < 0.5 else data_sample,
            'b': lambda: self._zero_out_person(data_sample) if num_people == 2 and np.random.rand() < 0.5 else data_sample,
            '1': lambda: tools.shear(data_sample, p=0.5),
            '2': lambda: tools.rotate(data_sample, p=0.5),
            '3': lambda: tools.scale(data_sample, p=0.5),
            '4': lambda: tools.spatial_flip(data_sample, p=0.5),
            '5': lambda: tools.temporal_flip(data_sample, frame_indices, p=0.5)[0],
            '6': lambda: tools.gaussian_noise(data_sample, p=0.5),
            '7': lambda: tools.gaussian_filter(data_sample, p=0.5),
            '8': lambda: tools.drop_axis(data_sample, p=0.5),
            '9': lambda: tools.drop_joint(data_sample, p=0.5),
        }
        for method, func in aug_methods.items():
            if method in self.aug_type:
                data_sample = func()

    def _inter_instance_augmentation(self, data_sample, frame_indices):
        """Apply inter-instance augmentation"""
        matching_indices = np.where(self.label == self.label[frame_indices])[0]
        match_data = np.array(self.data[random.choice(matching_indices)])
        match_frames = np.sum(match_data.sum(axis=(0, -1, -1)) != 0)
        frame_map = np.round((frame_indices + 1) * match_frames / 2).astype(int)
        match_data = match_data[:, frame_map]
        tools.skeleton_adain_bone_length(data_sample, match_data)

    def _zero_out_person(self, data_sample):
        """Zero out data of one person in multi-person samples"""
        axis_next = np.random.randint(2)
        data_sample[:, :, :, axis_next] = 0
        return data_sample

    def _convert_data_type(self, data_sample):
        """Convert data sample to specified data type"""
        if self.data_format == 'b':
            data_sample = tools.joint2bone()(data_sample)
        elif self.data_format == 'jm':
            data_sample = tools.to_motion(data_sample)
        elif self.data_format == 'bm':
            data_sample = tools.joint2bone()(data_sample)
            data_sample = tools.to_motion(data_sample)
        return data_sample

    def top_k_accuracy(self, score, top_k):
        """Calculate top-k accuracy"""
        ranked = score.argsort()
        return np.mean([l in ranked[i, -top_k:] for i, l in enumerate(self.label)])

# Function to dynamically import modules
def import_class(name):
    components = name.split('.')
    module = __import__(components[0])
    for comp in components[1:]:
        module = getattr(module, comp)
    return module
