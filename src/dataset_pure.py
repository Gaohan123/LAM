# Copyright 2021-2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""create train or eval dataset."""

import os
import warnings
import numpy as np
from typing import Callable, cast, Tuple

import mindspore as ms
import mindspore.dataset.engine as de
import mindspore.dataset.vision as C
import mindspore.dataset.transforms as C2
from mindspore.dataset.vision.utils import Inter
from mindspore.communication import get_rank, get_group_size

from .autoaugment import ImageNetPolicy

warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


class ToNumpy:
    def __init__(self):
        pass

    def __call__(self, img):
        return np.asarray(img)


def has_file_allowed_extension(filename: str, extensions: Tuple[str, ...]) -> bool:
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)


def make_dataset(directory, class_to_idx, extensions, is_valid_file):
    instances = []
    directory = os.path.expanduser(directory)
    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def is_valid_file(x: str) -> bool:
            return has_file_allowed_extension(x, cast(Tuple[str, ...], extensions))
    is_valid_file = cast(Callable[[str], bool], is_valid_file)
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    item = path, class_index
                    instances.append(item)
    return instances


class ImageFolder:
    def __init__(self, root, tran=None, name=None, extensions=IMG_EXTENSIONS, is_valid_file=None, seed=0):
        self.root = root
        classes, class_to_idx = self._find_classes(self.root)
        samples = self.make_dataset(self.root, class_to_idx, extensions, is_valid_file)
        if len(samples) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.root)
            raise RuntimeError(msg)
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]
        self._name = name
        self.tran = tran
        if self._name == 'random_no_fg':
            rng = np.random.RandomState(seed=seed)
            self._data_indices = rng.permutation(len(self.samples))

    @staticmethod
    def make_dataset(directory, class_to_idx, extensions, is_valid_file):
        return make_dataset(directory, class_to_idx, extensions=extensions, is_valid_file=is_valid_file)

    def _find_classes(self, dir):
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def __getitem__(self, index):
        if self._name == 'random_no_fg':
            index = self._data_indices[index]
        path, target = self.samples[index]
        sample = np.fromfile(path, np.uint8)

        if self.tran:
            sample = self.tran(sample)

        return sample, target

    def __len__(self):
        return len(self.samples)


# class JointImageDataset:
#     def __init__(self, train_dataset, pure_dataset, pure_data_ori, pure_data_ori_random, args):
#         self.train_dataset = train_dataset
#         self.pure_dataset = pure_dataset
#         self.pure_dataset_ori = pure_data_ori
#         self.pure_dataset_ori_random = pure_data_ori_random
#         self.seed = args.seed
#
#     def __len__(self):
#         return len(self.train_dataset)
#
#     def __getitem__(self, index):
#         image, label = self.train_dataset[index]
#         index_pure = index % len(self.pure_dataset)
#         pure_x, pure_y = self.pure_dataset[index_pure]
#
#         p = np.random.binomial(1, p=0.3)  # Probability of background replacement
#         if p == 1:
#             pure_bg_x, pure_ori_y = self.pure_dataset_ori[index_pure]
#         else:
#             pure_bg_x, pure_ori_y = self.pure_dataset_ori_random[index_pure]
#         pure_ori_x = pure_x.copy()
#         pure_ori_x[pure_x == 0] = pure_bg_x[pure_x == 0]
#
#         return image, label, pure_x, pure_y, pure_ori_x, pure_ori_y


class JointImageDataset:
    def __init__(self, train_dataset, pure_dataset, pure_data_ori, args):
        self.train_dataset = train_dataset
        self.pure_dataset = pure_dataset
        self.pure_dataset_ori = pure_data_ori
        self.seed = args.seed

    def __len__(self):
        return len(self.train_dataset)

    def __getitem__(self, index):
        image, label = self.train_dataset[index]
        index_pure = index % len(self.pure_dataset)
        pure_x, pure_y = self.pure_dataset[index_pure]
        pure_ori_x, pure_ori_y = self.pure_dataset_ori[index_pure]

        return image, label, pure_x, pure_y, pure_ori_x, pure_ori_y


def create_join_dataset(args,
                        image_size=224,
                        interpolation='BILINEAR',
                        crop_min=0.05,
                        repeat_num=1,
                        batch_size=32,
                        num_workers=12,
                        autoaugment=False):
    """create_dataset"""

    if hasattr(Inter, interpolation):
        interpolation = getattr(Inter, interpolation)
    else:
        interpolation = Inter.BILINEAR
        print('cannot find interpolation_type: {}, use {} instead'.format(interpolation, 'BILINEAR'))

    device_num = get_group_size()
    rank_id = get_rank()

    trans_ori = C.RandomCropDecodeResize(image_size, scale=(crop_min, 1.0),
                                         ratio=(0.75, 1.333), interpolation=interpolation)

    # foreground object uses a conservative cropping scale. Otherwise it may disappear in the mixed image,
    # a slight accuracy boost from this modification.
    trans_fg = C.RandomCropDecodeResize(image_size, scale=(0.9, 1.0),
                                        ratio=(0.75, 1.333), interpolation=interpolation)

    ds_train = ImageFolder(args.dataset_path, trans_ori, seed=args.seed)
    ds_pure = ImageFolder(args.pure_path, trans_fg, seed=args.seed)
    ds_pure_ori = ImageFolder(args.ori_path, trans_ori, seed=args.seed)
    # ds_pure_random = ImageFolder(args.no_fg_path, trans_ori, name='random_no_fg', seed=args.seed)

    type_cast_op = C2.TypeCast(ms.int32)

    # joint_dataset_generator = JointImageDataset(ds_train, ds_pure, ds_pure_ori, ds_pure_random, args)
    joint_dataset_generator = JointImageDataset(ds_train, ds_pure, ds_pure_ori, args)
    joint_dataset = de.GeneratorDataset(joint_dataset_generator, ["image", "label", "pure_x", "pure_y", "pure_ori_x",
                                                                  "pure_ori_y"], num_parallel_workers=num_workers,
                                        shuffle=True, num_shards=device_num, shard_id=rank_id)

    # Computed from random subset of ImageNet training images
    mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    std = [0.229 * 255, 0.224 * 255, 0.225 * 255]

    trans = [
        C.RandomHorizontalFlip(prob=0.5),
    ]
    if autoaugment:
        trans += [
            C.ToPIL(),
            ImageNetPolicy(),
            ToNumpy(),
        ]
    trans += [
        C.Normalize(mean=mean, std=std),
        C.HWC2CHW(),
    ]

    joint_dataset = joint_dataset.repeat(repeat_num)
    joint_dataset = joint_dataset.map(input_columns="image", num_parallel_workers=num_workers, operations=trans,
                                      python_multiprocessing=True)

    joint_dataset = joint_dataset.map(input_columns="pure_x", num_parallel_workers=num_workers, operations=trans,
                                      python_multiprocessing=True)
    joint_dataset = joint_dataset.map(input_columns="pure_ori_x", num_parallel_workers=num_workers, operations=trans,
                                      python_multiprocessing=True)
    joint_dataset = joint_dataset.map(input_columns="label", num_parallel_workers=num_workers, operations=type_cast_op)
    joint_dataset = joint_dataset.map(input_columns="pure_y", num_parallel_workers=num_workers, operations=type_cast_op)
    joint_dataset = joint_dataset.map(input_columns="pure_ori_y", num_parallel_workers=num_workers, operations=type_cast_op)

    joint_dataset = joint_dataset.batch(batch_size, drop_remainder=True)

    return joint_dataset
