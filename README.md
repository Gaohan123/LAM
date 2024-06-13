# Logit Attribution Matching (LAM)

## Introduction

This repository contains the PyTorch implementation of the paper *Consistency Regularization for Domain Generalization with Logit Attribution Matching*, accepted at UAI 2024.

Here, you will find code to execute ERM training, ERM with Copy-Paste augmentation, and LAM (Logit Attribution Matching) training with Copy-Paste augmentation on the iWildCam2020-WILDS dataset (iWildCam)

We have incorporated parts of the code for Copy-Paste augmentation from the [Targeted Augmentation project](https://github.com/i-gao/targeted-augs), and data handling code from the [WILDS repository](https://github.com/p-lambda/wilds).

## Dataset

The iWildCam dataset, which includes dataset splits, segmentation masks, and bounding boxes essential for the Copy-Paste augmentation, can be downloaded from this link: [iwildcam_data.zip](https://hkustconnect-my.sharepoint.com/:u:/g/personal/wxieai_connect_ust_hk/EUZoLIp5ZHtPhJ67X3F0hw0BdN-pZ1OWmT3FlBaOfwDUbA?e=wfKf4H).

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
| ERM      |     48.1           |   30.0                  | [ERM](https://hkustconnect-my.sharepoint.com/:u:/g/personal/wxieai_connect_ust_hk/ETidGoFRpn5IkHv_AIJhzNoBZnmezfwKG1MFF6Ygb7kfCA?e=Qfw1cC)|
| ERM+DA   |     53.8   |          36.0              |[ERM+DA](https://hkustconnect-my.sharepoint.com/:u:/g/personal/wxieai_connect_ust_hk/EeY6_sF4I75Hvy6L3QMMxgEBIZbZhop10gG75YNOK-aUNQ?e=KrN7BW)|
| LAM      |     52.6          |   42.3                  |[LAM](https://hkustconnect-my.sharepoint.com/:u:/g/personal/wxieai_connect_ust_hk/EcO6ybPmm-ZPo49ZyupogEkBRvfNnbWk2coBxyPburPVxA?e=UEyOcp)|

### Evaluation
```bash
python evaluation.py \
--config=configs/lam_wildcam.yaml \
--log_dir=evaluation_log \
--root_dir=iwildcam_data\
--gpu_id=0\
--checkpoint_path=lam_checkpoints/ERM
```


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
