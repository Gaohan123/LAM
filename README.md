# Logit Attribution Matching (LAM)

## Introduction

This repository contains the PyTorch implementation of the paper [*Consistency Regularization for Domain Generalization with Logit Attribution Matching*](https://openreview.net/pdf?id=WNy1ooHYHx), accepted at UAI 2024.

Here, you will find code to execute ERM training, ERM with Copy-Paste augmentation, and LAM (Logit Attribution Matching) training with Copy-Paste augmentation on the iWildCam2020-WILDS dataset (iWildCam)

We have incorporated parts of the code for Copy-Paste augmentation from the [Targeted Augmentation project](https://github.com/i-gao/targeted-augs), and data handling code from the [WILDS repository](https://github.com/p-lambda/wilds).

## Dataset

The iWildCam dataset, which includes dataset splits, segmentation masks, and bounding boxes essential for the Copy-Paste augmentation, can be downloaded from this link: [iwildcam_data.zip]().

Please download and extract the contents to retrieve the `iwildcam_data` folder.

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
|----------|----------------------|-----------------------------|----------|
| ERM      |                |                     ||
| ERM+DA   |         |                        ||
| LAM      |                |                        ||

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
