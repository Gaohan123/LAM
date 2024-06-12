# Logit Attribution Matching (LAM)

## Introduction

This repository is the PyTorch implementation of <em> Consistency Regularization for Domain Generalization with Logit Attribution Matching </em>, which is accepted by UAI 2024.

The codes are released to re-implement the ERM training, ERM training with Copy-Paste augmented examples, and LAM training with Copy-Paste augmented examples on the iWildCam2020-WILDS (iWildCam) dataset.

Part of the codes related to the Copy-Paste augmentation were adopted from the released codes of [Targeted Augmentation](https://github.com/i-gao/targeted-augs), and the data-related codes were adopted from [WILDS](https://github.com/p-lambda/wilds).

## Dataset

The iWildCam dataset, including the split of the dataset, and the segmentation masks and bounding boxes for the training images (which are used in Copy-Paste augmentation), are available in this [iwildcam_data.zip]().

After downloading it, please unzip it.

## Code
To install dependencies, run
```
pip install -r requirements.txt
```

### ERM Training
```bash
python train_erm.py \
--config=configs/lam_wildcam.yaml \
--log_dir=erm_log \
--root_dir=iwildcam_data\
--gpu_id=0
```

### ERM Training with Copy-Paste augmentation
```bash
python train_erm_cp.py \
--config=configs/lam_wildcam.yaml \
--log_dir=erm_cp_log \
--root_dir=iwildcam_data\
--gpu_id=0
```

### LAM Training with Copy-Paste augmentation
```bash
python train_lam_cp.py \
--config=configs/lam_wildcam.yaml \
--log_dir=lam_cp_log \
--root_dir=iwildcam_data\
--gpu_id=0
```

## Result

The result on the ID and OOD testing set  is shown below.
| Model    | ID Macro F1 score | OOD Macro F1 score | Model Checkpoint    |
|----------|----------------------|-----------------------------|
| ERM      |                |                     |
| ERM+DA   |         |                        |
| LAM      |                |                        |

## Citation
If this codebase / these models are useful in your work, please consider citing our paper.

```
@inproceedings{
gao2024consistency,
title={Consistency Regularization for Domain Generalization with Logit Attribution Matching},
author={Han Gao and Kaican Li and Weiyan Xie and Zhi LIN and Yongxiang Huang and Luning Wang and Caleb Chen Cao and Nevin L. Zhang},
booktitle={The 40th Conference on Uncertainty in Artificial Intelligence},
year={2024},
url={https://openreview.net/forum?id=WNy1ooHYHx}
}
```

## Contact
Weiyan Xie via wxieai@cse.ust.hk
