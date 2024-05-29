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
"""loss functions"""

import mindspore as ms
import mindspore.ops as ops
from mindspore import nn, Parameter
from mindspore import Tensor
from mindspore.nn.loss.loss import LossBase as Loss


class CrossEntropySmooth(Loss):
    """CrossEntropy"""
    def __init__(self, sparse=True, reduction='mean', smooth_factor=0., num_classes=1000, aux_factor=0.4):
        super().__init__()
        self.aux_factor = aux_factor
        self.onehot = ops.OneHot()
        self.sparse = sparse
        self.on_value = Tensor(1.0 - smooth_factor, ms.float32)
        self.off_value = Tensor(1.0 * smooth_factor / (num_classes - 1), ms.float32)
        self.ce = nn.SoftmaxCrossEntropyWithLogits(reduction=reduction)

    def construct(self, logits, label):
        if isinstance(logits, tuple):
            logit, aux_logit = logits
        else:
            logit, aux_logit = logits, None

        if self.sparse:
            label = self.onehot(label, logit.shape[1], self.on_value, self.off_value)

        loss = self.ce(logit, label)
        if aux_logit is not None:
            loss = loss + self.aux_factor * self.ce(aux_logit, label)
        return loss


class CrossEntropySimple(Loss):
    """CrossEntropy"""
    def __init__(self, reduction='mean'):
        super().__init__()
        self.cross_entropy = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction=reduction)

    def construct(self, logit, label):
        loss = self.cross_entropy(logit, label)
        return loss


class CrossEntropyIgnore(Loss):
    """CrossEntropyIgnore"""
    def __init__(self, num_classes=21, ignore_label=255):
        super().__init__()
        self.one_hot = ops.OneHot(axis=-1)
        self.on_value = Tensor(1.0, ms.float32)
        self.off_value = Tensor(0.0, ms.float32)
        self.cast = ops.Cast()
        self.ce = nn.SoftmaxCrossEntropyWithLogits()
        self.not_equal = ops.NotEqual()
        self.num_cls = num_classes
        self.ignore_label = ignore_label
        self.mul = ops.Mul()
        self.sum = ops.ReduceSum(False)
        self.div = ops.RealDiv()
        self.transpose = ops.Transpose()
        self.reshape = ops.Reshape()

    def construct(self, logits, labels):
        labels_int = self.cast(labels, ms.int32)
        labels_int = self.reshape(labels_int, (-1,))
        logits_ = self.transpose(logits, (0, 2, 3, 1))
        logits_ = self.reshape(logits_, (-1, self.num_cls))
        weights = self.not_equal(labels_int, self.ignore_label)
        weights = self.cast(weights, ms.float32)
        one_hot_labels = self.one_hot(labels_int, self.num_cls, self.on_value, self.off_value)
        loss = self.ce(logits_, one_hot_labels)
        loss = self.mul(weights, loss)
        loss = self.div(self.sum(loss), self.sum(weights))
        return loss


class CustomWithLossCell(nn.Cell):
    def __init__(self, backbone, batch_size=256, epochs=1, step_size=1, num_classes=1000):
        super(CustomWithLossCell, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self.cross_entropy = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
        self.num_classes = num_classes
        self.expand_dims = ops.ExpandDims()
        self.sum = ops.ReduceSum()
        self.epochs = epochs
        self.step_size = step_size
        self.current_step = Parameter(Tensor(0, ms.float32), requires_grad=False)
        self.lam = 1.0  # constant lambda coefficient
        self.batch_size = batch_size

    def construct(self, images, labels, pure_x, pure_y, pure_ori_x, pure_ori_y):
        # pure_y and pure_ori_y is expected to be same.
        outputs, _ = self._backbone(images)
        _, cfeatures_pure = self._backbone(pure_x)
        _, cfeatures_pure_ori = self._backbone(pure_ori_x)

        weight_fc = self._backbone.head_new.weight
        diff_sum_all = Tensor(0, ms.float32)

        for label in range(self.num_classes):
            index_label = pure_y == label
            index_label = index_label.reshape(-1, 1)

            f_pure = cfeatures_pure.copy()
            f_ori = cfeatures_pure_ori.copy()

            f_pure *= index_label
            f_ori *= index_label

            diff_feature_2 = (f_pure - f_ori) ** 2
            weight_term = weight_fc[label] ** 2

            weight_term = self.expand_dims(weight_term, 1)
            diff_sum = ops.matmul(diff_feature_2, weight_term)

            diff_sum_all = diff_sum_all + self.sum(diff_sum)

        diff_sum_all = 1.0 * diff_sum_all / self.batch_size
        current_epoch = ops.floor_div(self.current_step, self.step_size)
        current_epoch = current_epoch.astype(ms.int32)

        loss = self.cross_entropy(outputs, labels) + self.lam * diff_sum_all
        self.current_step += 1

        return loss


def get_loss(loss_name, args):
    """get_loss"""
    loss = None
    if loss_name == 'ce_smooth':
        loss = CrossEntropySmooth(smooth_factor=args.label_smooth_factor,
                                  num_classes=args.class_num,
                                  aux_factor=args.aux_factor)
    elif loss_name == 'ce_simple':
        loss = CrossEntropySimple()
    elif loss_name == 'ce_ignore':
        loss = CrossEntropyIgnore(num_classes=args.class_num,
                                  ignore_label=args.ignore_label)
    else:
        raise NotImplementedError

    return loss
