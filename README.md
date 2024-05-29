# Logit Attribution Matching (LAM) (Mindspore)

## Introduction

This repository is the MindSpore implementation of <em> Consistency Regularization for Domain Generalization with Logit Attribution Matching </em>, which is accepted by UAI 2024.

## Get Started
It supports LAM with pretrained ViT-patch32 model[1] based on MindSpore and Ascend platforms.

### Preparation

#### Dependency
- mindspore >= 2.3.0.rc2  [[install](https://www.mindspore.cn/install)]
- python >= 3.8.18

Install the dependent packages by running:
```shell
pip install -r requirements.txt
```

#### Pretrained Models
Please download the pretrained ViT-patch32 model from [[Mindspore Modelzoo](https://download.mindspore.cn/models/r1.9/vit_ascend_v190_imagenet2012_official_cv_acc73.81.ckpt)]

#### Data preparation
Please download the ImageNet-9 dataset and its variations from [[LAM datasets](https://drive.google.com/drive/folders/1oB3OSH23A_A0o1seXrIep9Wq2zNerJt1?usp=drive_link)] and [[ImageNet-9](https://github.com/MadryLab/backgrounds_challenge)]. They includes:

- original: 9 coarse classes of images fetched from ImageNet dataset. It contains training and validation samples. We use it to do ERM fintuning.
- only_fg_few005: 9 coarse classes of images. Each of them only has foreground objects while the background is removed by Grabcut. And we randomly sampled 5% number of images compared with the amount of original. We use it to combine with original images as pairs to do LAM finetuning.
- original_few005: 9 coarse classes of images. It contains the counterpart origianl images in only_fg_few005. We use it to combine with only_fg images as pairs to do LAM finetuning.
- original_fg_few005: 9 coarse classes of images. Based on original, it adds the images in only_fg_few005 as augmented samples directly. We use it in ERM+DA finetuning and LAM finetuning.
- mixed_rand: 9 coarse classes of images. For each image, it has foreground of an original image placed onto the background of a random image. We use its validation set as the ood test domain.


## ERM finetuning
1. In `config/vit_patch32_imagenet9_erm_lp.yml`, change `dataset_path`, `eval_path` to the training and validation folder of ImageNet-9 original sepearetely; change `pretrained` to the path of pratrained model checkpoint downloaded from [[Mindspore Modelzoo](https://download.mindspore.cn/models/r1.9/vit_ascend_v190_imagenet2012_official_cv_acc73.81.ckpt)];

2. Start finetuning (linear prob) by 
```shell script
bash scripts/run_train_erm_lp_ascend.sh
```
check training log in `./device_erm_lp`

3. Modify `config/vit_patch32_imagenet9_erm_ft.yml` as same as step 1. change `pretrained` to the checkpoint path of best lp finetuning `./best_erm_lp`.

4. Start finetuning(FT) by
```shell script
bash scripts/run_train_erm_ft_ascend.sh
```
check training log in `./device_erm_ft`

5. The best checkpoint in `./best_erm_ft` will be used for validation set / testing set (OOD) evaluation.

## ERM+TargetedDA finetuning
1. In `config/vit_patch32_imagenet9_erm_da_lp.yml`, change `dataset_path`, `eval_path` to the training and validation folder of ImageNet-9 original_fg_few005 sepearetely; change `pretrained` to the path of pratraind model checkpoint downloaded from [[Mindspore Modelzoo](https://download.mindspore.cn/models/r1.9/vit_ascend_v190_imagenet2012_official_cv_acc73.81.ckpt)];

2. Start finetuning (linear prob) by 
```shell script
bash scripts/run_train_erm_da_lp_ascend.sh
```
check training log in `./device_erm_da_lp`

3. Modify `config/vit_patch32_imagenet9_erm_da_ft.yml` as same as step 1. change `pretrained` to the path of best lp finetuning `./best_erm_da_lp`.

4. Start finetuning(FT) by
```shell script
bash scripts/run_train_erm_da_ft_ascend.sh
```
check training log in `./device_erm_da_ft`

5. The best checkpoint in `./best_erm_da_ft` will be used for validation set / testing set (OOD) evaluation.

## LAM finetuning
1. In `config/vit_patch32_imagenet9_lam_lp.yml`, change `dataset_path`, `eval_path` to the training and validation folder of ImageNet-9 original_fg_few005 sepearetely; change `pretrained` to the path of pratraind model checkpoint downloaded from [[Mindspore Modelzoo](https://download.mindspore.cn/models/r1.9/vit_ascend_v190_imagenet2012_official_cv_acc73.81.ckpt)]; change `pure_path` to the folder only_fg_few005; change `ori_path` to the folder contains original_few005.

2. Start finetuning (linear prob) by 
```shell script
bash scripts/run_train_lam_lp_ascend.sh
```
check training log in `./device_lam_lp`

3. Modify `config/vit_patch32_imagenet9_lam_ft.yml` as same as step 1. Change `pretrained` to the path of best lp finetuning `./best_lam_lp`.

4. Start finetuning(FT) by
```shell script
bash scripts/run_train_lam_ft_ascend.sh
```
check training log in `./device_lam_ft`

5. The best checkpoint in `./best_lam_ft` will be used for validation set / testing set (OOD) evaluation.

## model evaluation
1. In `config/vit_eval.yml`, change `eval_path` to the validation set of mixed-rand; change `pretrained` to the checkpoints saved in `./best_xxx`. 

2. Start model evaluation by
```shell script
bash scripts/run_eval_ascend.sh
```

3. The top-1 accuracy is shown in `./device_eval` file.
## Result

The result on validation set (imagenet_9) and testing set (mixed_rand) is shown below.
| Model    | Top-1 Accuracy (val) | Top-1 Accuracy (test - OOD) |
|----------|----------------------|-----------------------------|
| ERM      | 97.68                | 72.90                       |
| ERM+TargetedDA   | 97.68        | 75.20                       |
| LAM      | 96.63                | 76.56                       |


## Reference
[1] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby. 2021.