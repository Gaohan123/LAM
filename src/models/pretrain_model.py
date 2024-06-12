
import torch
from torch import nn
import os
from . import model_utils
from torchvision.transforms import Normalize
from torchvision.models import resnet50


normalize_transform = Normalize(
    mean=(0.48145466, 0.4578275, 0.40821073),
    std=(0.26862954, 0.26130258, 0.27577711))


class Model(nn.Module):
    def __init__(self, model_name, scratch=False):
        super().__init__()
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        model = resnet50(pretrained=True)
        self._num_in_feature=model.fc.in_features
        model.fc = nn.Sequential()
        model = model.eval().to(self._device)
        self._model = model
        self._classifier = None
    def forward(self, x):
        features = self.get_features(x)
        if self._classifier is None:
            return features
        return self._classifier(features), features
    def set_requires_grad(self, val):
        for param in self._model.parameters():
            param.requires_grad = val
        if self._classifier is not None:
            for param in self._classifier.parameters():
                param.requires_grad = val
    def new_last_layer(self, num_classes):
        num_in_features = self._num_in_feature
        self._classifier = nn.Linear(num_in_features, num_classes)
        self._classifier.to(self._device)
    def add_probe(self, probe):
        self._classifier = probe
    def get_last_layer(self):
        return self._classifier
    def set_last_layer(self, coef, intercept):
        set_linear_layer(self._classifier, coef, intercept)
    def get_feature_extractor(self):
        raise NotImplementedError('Be careful, we need to normalize image first before encoding it.')
    def get_features(self, x):
        return self._model(normalize_transform(x))
