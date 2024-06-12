

import os
import os.path
import sys
import torch
from torchvision import transforms
import torch.backends.cudnn as cudnn
import argparse
import json
import logging
import os
import sys
import random
from PIL import Image
import numpy as np


def isin(elements, test_elements):
    """return if elements are in test_elements"""
    if torch.is_tensor(elements): elements = elements.numpy()
    if torch.is_tensor(test_elements): test_elements = test_elements.numpy()
    mask = np.isin(elements, test_elements)
    return torch.tensor(mask)



def get_empty_img_by_y(dataset,y):
    labeled_mask = isin(dataset.location_array, dataset.y_to_observed_locs[y])[dataset.empty_indices]
    return sample_empty_img_given_mask(dataset,labeled_mask)


def sample_empty_img_given_mask(dataset,labeled_mask):
    """return a random item from empty_indices that satisfies the given boolean masks"""
    assert len(labeled_mask) == len(dataset.empty_indices)
    # empty indices are numpy arrays, so make masks numpy
    labeled_mask = labeled_mask.numpy()
    if np.all(labeled_mask == 0):
        return None
    empty_ix = np.random.choice(dataset.empty_indices[labeled_mask])
    bg_image_name=dataset._input_array[empty_ix]
    return bg_image_name

def get_nonempty_img_by_y(dataset,y):
    labeled_mask = isin(dataset.location_array, dataset.y_to_observed_locs[y])[dataset.nonempty_indices]
    return sample_nonempty_img_given_mask(dataset,labeled_mask)


def sample_nonempty_img_given_mask(dataset,labeled_mask):
    """return a random item from nonempty_indices that satisfies the given boolean masks"""
    assert len(labeled_mask) == len(dataset.nonempty_indices)
    # nonempty indices are numpy arrays, so make masks numpy
    labeled_mask = labeled_mask.numpy()
    if np.all(labeled_mask == 0):
        return None
    nonempty_ix = np.random.choice(dataset.nonempty_indices[labeled_mask])
    bg_image_name=dataset._input_array[nonempty_ix]
    return bg_image_name



transform_bg= transforms.Compose([
    transforms.Resize((448,448)),
    transforms.ToTensor(),
])


def img_to_tensor_bg(img_path):
    image = Image.open(img_path).convert('RGB')
    input_ = transform_bg(image)
    return input_
