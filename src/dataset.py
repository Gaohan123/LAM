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

import warnings
import numpy as np

import mindspore as ms
from mindspore import ops
import mindspore.dataset.engine as de
import mindspore.dataset.vision as C
import mindspore.dataset.transforms as C2
from mindspore.dataset.vision.utils import Inter
from mindspore.communication import get_rank, get_group_size

from .autoaugment import ImageNetPolicy

warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)


class ToNumpy:
    def __init__(self):
        pass

    def __call__(self, img):
        return np.asarray(img)


class JointImageDataset:
    def __init__(self, pure_dataset, pure_data_ori, pure_data_ori_random, args):
        self.pure_dataset = pure_dataset
        self.pure_dataset_ori = pure_data_ori
        self.pure_dataset_ori_random = pure_data_ori_random

        self.pure_dataset_iter = pure_dataset.create_tuple_iterator()
        self.pure_dataset_ori_iter = pure_data_ori.create_tuple_iterator()
        self.pure_dataset_ori_random_iter = pure_data_ori_random.create_tuple_iterator()
        self.seed = args.seed
        self.uniform = ops.UniformReal(seed=3)

    def __len__(self):
        return min(self.pure_dataset.get_dataset_size(), self.pure_dataset_ori.get_dataset_size())

    def __getitem__(self, index):
        pure_x, pure_y = self.pure_dataset_iter[index]
        p = self.uniform((1,))[0]

        if p > 0.7:
            pure_bg_x, pure_ori_y = self.pure_dataset_ori_iter[index]
        else:
            pure_bg_x, pure_ori_y = self.pure_dataset_ori_random_iter[index]
        pure_ori_x = pure_x.copy()
        pure_ori_x[pure_x == 0] = pure_bg_x[pure_x == 0]

        return pure_x, pure_y, pure_ori_x, pure_ori_y


def create_join_dataset(pure_path,
                        no_fg_path,
                        args,
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

    ds_pure = de.ImageFolderDataset(pure_path, num_parallel_workers=num_workers, shuffle=False,
                                    num_shards=device_num, shard_id=rank_id)

    ds_pure_ori = de.ImageFolderDataset(no_fg_path, num_parallel_workers=num_workers, shuffle=False,
                                        num_shards=device_num, shard_id=rank_id)

    ds_pure_random = de.ImageFolderDataset(no_fg_path, num_parallel_workers=num_workers, shuffle=True,
                                           num_shards=device_num, shard_id=rank_id)

    trans_ori = [
        C.RandomCropDecodeResize(image_size, scale=(crop_min, 1.0), ratio=(0.75, 1.333), interpolation=interpolation),
    ]

    type_cast_op = C2.TypeCast(ms.int32)

    ds_pure = ds_pure.repeat(repeat_num)
    ds_pure = ds_pure.map(input_columns="image", num_parallel_workers=num_workers, operations=trans_ori,
                          python_multiprocessing=True)
    ds_pure = ds_pure.map(input_columns="label", num_parallel_workers=num_workers, operations=type_cast_op)

    ds_pure_ori = ds_pure_ori.repeat(repeat_num)
    ds_pure_ori = ds_pure_ori.map(input_columns="image", num_parallel_workers=num_workers, operations=trans_ori,
                                  python_multiprocessing=True)
    ds_pure_ori = ds_pure_ori.map(input_columns="label", num_parallel_workers=num_workers, operations=type_cast_op)

    ds_pure_random = ds_pure_random.repeat(repeat_num)
    ds_pure_random = ds_pure_random.map(input_columns="image", num_parallel_workers=num_workers, operations=trans_ori,
                                        python_multiprocessing=True)
    ds_pure_random = ds_pure_random.map(input_columns="label", num_parallel_workers=num_workers, operations=type_cast_op)

    joint_dataset_generator = JointImageDataset(ds_pure, ds_pure_ori, ds_pure_random, args)
    joint_dataset = de.GeneratorDataset(joint_dataset_generator, ["pure_x", "pure_y", "pure_ori_x", "pure_ori_y"],
                                        num_parallel_workers=num_workers, shuffle=True, num_shards=device_num,
                                        shard_id=rank_id)

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
    joint_dataset = joint_dataset.map(input_columns="pure_x", num_parallel_workers=num_workers, operations=trans,
                                      python_multiprocessing=True)
    joint_dataset = joint_dataset.map(input_columns="pure_ori_x", num_parallel_workers=num_workers, operations=trans,
                                      python_multiprocessing=True)
    joint_dataset = joint_dataset.map(input_columns="pure_y", num_parallel_workers=num_workers, operations=type_cast_op)
    joint_dataset = joint_dataset.map(input_columns="pure_ori_y", num_parallel_workers=num_workers, operations=type_cast_op)

    joint_dataset = joint_dataset.batch(batch_size, drop_remainder=True)

    return joint_dataset


def create_dataset(dataset_path,
                   do_train,
                   image_size=224,
                   interpolation='BILINEAR',
                   crop_min=0.05,
                   repeat_num=1,
                   batch_size=32,
                   num_workers=12,
                   autoaugment=False,
                   mixup=0.0,
                   num_classes=1001):
    """create_dataset"""

    if hasattr(Inter, interpolation):
        interpolation = getattr(Inter, interpolation)
    else:
        interpolation = Inter.BILINEAR
        print('cannot find interpolation_type: {}, use {} instead'.format(interpolation, 'BILINEAR'))

    device_num = get_group_size()
    rank_id = get_rank()

    if do_train:
        ds = de.ImageFolderDataset(dataset_path, num_parallel_workers=num_workers, shuffle=True,
                                   num_shards=device_num, shard_id=rank_id)
    else:
        ds = de.ImageFolderDataset(dataset_path, num_parallel_workers=num_workers,
                                   shuffle=False, num_shards=device_num, shard_id=rank_id)
        print("eval dataset size: {}".format(ds.get_dataset_size()))

    # Computed from random subset of ImageNet training images
    mean = [0.485*255, 0.456*255, 0.406*255]
    std = [0.229*255, 0.224*255, 0.225*255]

    # define map operations
    if do_train:
        trans = [
            C.RandomCropDecodeResize(image_size, scale=(crop_min, 1.0),
                                     ratio=(0.75, 1.333), interpolation=interpolation),
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
    else:
        resize = int(int(image_size / 0.875 / 16 + 0.5) * 16)
        print('eval, resize:{}'.format(resize))
        trans = [
            C.Decode(),
            C.Resize(resize, interpolation=interpolation),
            C.CenterCrop(image_size),
            C.Normalize(mean=mean, std=std),
            C.HWC2CHW()
        ]

    type_cast_op = C2.TypeCast(ms.int32)

    ds = ds.repeat(repeat_num)
    ds = ds.map(input_columns="image", num_parallel_workers=num_workers, operations=trans, python_multiprocessing=True)
    ds = ds.map(input_columns="label", num_parallel_workers=num_workers, operations=type_cast_op)

    if do_train and mixup > 0:
        one_hot_encode = C2.OneHot(num_classes)
        ds = ds.map(operations=one_hot_encode, input_columns=["label"])

    ds = ds.batch(batch_size, drop_remainder=True)

    if do_train and mixup > 0:
        trans_mixup = C.MixUpBatch(alpha=mixup)
        ds = ds.map(input_columns=["image", "label"], num_parallel_workers=num_workers, operations=trans_mixup)

    return ds


def get_dataset(dataset_name, do_train, dataset_path, args):
    if dataset_name == "imagenet":
        if do_train:
            data = create_dataset(dataset_path=dataset_path,
                                  do_train=True,
                                  image_size=args.train_image_size,
                                  interpolation=args.interpolation,
                                  autoaugment=args.autoaugment,
                                  mixup=args.mixup,
                                  crop_min=args.crop_min,
                                  batch_size=args.batch_size,
                                  num_workers=args.train_num_workers,
                                  num_classes=args.class_num)
        else:
            data = create_dataset(dataset_path=dataset_path,
                                  do_train=False,
                                  image_size=args.eval_image_size,
                                  interpolation=args.interpolation,
                                  batch_size=args.eval_batch_size,
                                  num_workers=args.eval_num_workers,
                                  num_classes=args.class_num)
    elif dataset_name == "cifar10":
        if do_train:
            data = create_dataset(dataset_path=dataset_path,
                                  do_train=True,
                                  image_size=args.train_image_size,
                                  interpolation=args.interpolation,
                                  autoaugment=args.autoaugment,
                                  mixup=args.mixup,
                                  crop_min=args.crop_min,
                                  batch_size=args.batch_size,
                                  num_classes=10,
                                  num_workers=args.train_num_workers)
        else:
            data = create_dataset(dataset_path=dataset_path,
                                  do_train=False,
                                  image_size=args.eval_image_size,
                                  interpolation=args.interpolation,
                                  batch_size=args.eval_batch_size,
                                  num_classes=10,
                                  num_workers=args.eval_num_workers)
    else:
        raise NotImplementedError
    return data
