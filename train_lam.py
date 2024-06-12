import os
import os.path
import sys
import torch
from torchvision import transforms
import torch.backends.cudnn as cudnn
import argparse
import json
import logging
import pandas as pd
from pathlib import Path
import pickle
import shutil
import random
import quinine
import numpy as np

from src.utils.accumulator import Accumulator
import src.utils.utils as utils
from src.models.model_utils import *
from utils import *
from eval_utils import *

from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader
import torchvision.transforms as transforms
from wilds.common.grouper import CombinatorialGrouper

from copy_paste_aug.select_bg import *
from copy_paste_aug.wild_mask_bbox import *

import ssl
ssl._create_default_https_context = ssl._create_unverified_context
log_level = logging.INFO


def copy_paste_augmentation(args, inputs, labels, img_ids, index_list, mask_dir, bbox_df, dataset, mask_img_ids, bbox_img_ids):
    # create_pair_num11 is the index of image where the augmented examples can be created successfully
    create_pair_num11 = []
    augmented_x = []
    augmented_y = []
    num1 = inputs.shape[0]
    for num11 in range(num1):
        label1 = labels[num11].item()
        if label1 != 0:
            image_name = img_ids[index_list[num11].item()]
            # generate_final_mask: to get the segmentation mask information for the image
            # if the segmentation mask for the image is available: return the segmentation masks
            # elif the bounding box for the image is available: return the segmentation masks induced from the bounding bbox
            # else: return None
            mask, mask_value = generate_final_mask(mask_dir, bbox_df, image_name, mask_img_ids, bbox_img_ids)
            if mask is not None:
                # get_empty_img_by_y: to find an empty background image where the animal has be historically identified
                # if there are such background images, randomly select one of them and return the image name
                # if there is no such background image, return None.
                bg_image_name = get_empty_img_by_y(dataset, label1)
                if bg_image_name is not None:
                    bg_image_path = args.root_dir + '/iwildcam_v2.0/train/' + str(bg_image_name)
                    input_bg = img_to_tensor_bg(bg_image_path).cuda()
                    mask_value1 = np.asarray([mask_value, mask_value, mask_value])
                    mm = np.expand_dims((mask_value1 == True), axis=0)
                    input_bg[mm] = inputs[num11][mm]
                    input_bg = transforms.RandomHorizontalFlip()(input_bg)
                    augmented_x.append(input_bg)
                    augmented_y.append(label1)
                    create_pair_num11.append(num11)
    return augmented_x, augmented_y, create_pair_num11


def train(args, mask_dir, bbox_df, dataset, mask_img_ids, bbox_img_ids, img_ids, epoch, config, train_loader,  net, device, optimizer, criterion,
          weight_dict_initial, lam_item, is_ft=False):
    """
    mask_dir: the directory that stores the segmentation masks for the iWildCam training image
    bbox_df: the dataframe that stores the bounding box information for the iWildCam training image
    dataset: the iWildCam dataset class
    mask_img_ids: the ordered list of image names in the mask directory
    bbox_img_ids: the ordered list of image names in the bounding box dataframe
    img_ids: the ordered list of image names in the iWildCam dataset
    """
    # Returns a dictionary with epoch, train/loss and train/acc.
    # Train model.
    training_state = net.training
    logging.info("\nEpoch #{}".format(epoch))
    loss_dict = {
        'train/loss': Accumulator(),
        'train/acc': Accumulator(),
        'train/f1': Accumulator(),
    }

    num_examples = 0
    for i, data in enumerate(train_loader, 0):
        net.train()
        if config['use_cuda']:
            data = utils.to_device(data, device)

        inputs, labels,_,index_list = data

        ### Conduct the Copy-and-Paste Augmentation
        # create_pair_num11 is the index of image where the augmented examples can be created successfully
        augmented_x, augmented_y, create_pair_num11 = copy_paste_augmentation(args, inputs, labels, img_ids, index_list,
        mask_dir,bbox_df, dataset, mask_img_ids, bbox_img_ids)

        ### START: Logit Attribution Matching (LAM) Implementation
        if len(augmented_x)>0:
            augmented_x = torch.stack((augmented_x))
            augmented_x = augmented_x.cuda()
            augmented_y=torch.from_numpy(np.asarray(augmented_y)).cuda()
            optimizer.zero_grad()
            input_len = inputs.shape[0]
            augmented_len = augmented_x.shape[0]
            all_x = torch.cat([inputs, augmented_x])
            labels = torch.cat([labels, augmented_y])
            outputs, cfeatures = net(all_x)
            cfeatures_augmented = cfeatures[input_len: ]
            cfeatures_ori = cfeatures[create_pair_num11]

            if config['linear_probe']:
                cfeatures_ori = cfeatures_ori.detach()
                cfeatures_augmented = cfeatures_augmented.detach()

            weight_fc = net._classifier.weight
            diff_sum_all = 0
            sample_num = 0

            for label in range(outputs.shape[1]):
                f_augmented = cfeatures_augmented[torch.where(augmented_y == label)[0]]
                f_ori = cfeatures_ori[torch.where(augmented_y == label)[0]]
                diff_feature = (f_augmented - f_ori) ** 2
                weight_term = weight_fc[label] ** 2
                weight_term = torch.unsqueeze(weight_term, 1)
                diff_sum = torch.matmul(diff_feature, weight_term)
                diff_sum_all = diff_sum_all + torch.sum(diff_sum)
                sample_num = sample_num + f_augmented.shape[0]

            diff_sum_all = 1.0 * diff_sum_all / sample_num
            loss_erm = criterion(outputs, labels)
            loss_lam = diff_sum_all
            loss = loss_erm + config['lam_item'] * loss_lam
        ### END: Logit Attribution Matching (LAM) Implementation

        ### IF not suitable augmentation examples are generated for all images in the batch,
        ### Now LAM reduces to ERM
        if len(augmented_x)==0:
            all_x = inputs
            labels =labels
            optimizer.zero_grad()
            outputs, cfeatures = net(all_x)
            loss_erm = criterion(outputs, labels)
            loss_lam = 0
            loss = loss_erm + config['lam_item'] * loss_lam

        _, train_preds = torch.max(outputs.data, axis=1)
        loss_dict['train/loss'].add_value(loss.tolist())
        loss_dict['train/acc'].add_values((train_preds == labels).tolist())
        y_true=labels.detach().cpu().numpy()
        y_pred=train_preds.cpu().numpy()
        f1=f1_score(y_true, y_pred, average='macro', labels=np.unique(y_true))
        f1_list = f1.tolist()
        loss_dict['train/f1'].add_value(f1_list)

        if i%100==0:
            list111=(train_preds == labels).tolist()
            acc111=np.sum(list111)/len(list111)
            y_true=labels.detach().cpu().numpy()
            y_pred=train_preds.cpu().numpy()
            f1=f1_score(y_true, y_pred, average='macro', labels=np.unique(y_true))
            print('epoch:'+str(epoch)+'-'+str(i)+'/'+str(len(train_loader)))
            print('loss_erm:'+str(loss_erm)+';'+'loss_lam:'+str(loss_lam)+';'+'loss:'+str(loss))
            print('training accuracy:'+str(acc111))
            print('macro-f1:'+str(f1))

        loss.backward()
        optimizer.step()
        num_examples += len(labels)
        outputs, loss, train_preds = None, None, None  # Try to force garbage collection.
        def should_log(log_interval):
            return num_examples // log_interval > (num_examples - len(labels)) // log_interval
        if should_log(config['log_interval']):
            for k in loss_dict:
                logging.info(
                    '[%d, %5d] %s: %.3f' %
                    (epoch + 1, num_examples, k, loss_dict[k].get_mean()))

    reset_state(net, training_state)
    train_stats = {'epoch': epoch}
    for key in loss_dict:
        train_stats[key] = loss_dict[key].get_mean()
    return train_stats



def main(args,config, log_dir, checkpoints_dir):
    """
    Main function
    """
    # Set up datasets and loaders.
    print("Entering main.")
    logging.info("Entering main.")

    ## Define the iWildCam training dataset
    dataset = get_dataset(dataset="iwildcam",  root_dir=args.root_dir,download=False)
    train_data = dataset.get_subset(
        "train",
        transform=transforms.Compose(
            [transforms.Resize((448, 448)), transforms.ToTensor()]
        ),)
    grouper = CombinatorialGrouper(dataset, ['location'])
    ## Define the location of segmentation masks and empty background image for iWildCam
    dataset.empty_indices = dataset.empty_indices['train']
    root_dir=args.root_dir
    data_dir=str(root_dir)+'/'+'iwildcam_v2.0/'
    mask_dir = os.path.join(data_dir, 'instance_masks')
    bbox_path = os.path.join(data_dir, 'megadetector_results.json')
    bbox_df = pd.DataFrame(json.load(open(bbox_path, 'r'))['images']).set_index('id')
    df = pd.read_csv(data_dir+ 'metadata.csv')
    img_ids = df['image_id']
    bbox_img_ids=list(bbox_df.index)
    mask_img_ids=[]
    for i in os.listdir(mask_dir):
        mask_img_ids.append(i.split('.')[0])

    # Set up image loaders.
    train_loader = get_train_loader("standard", train_data, batch_size=config['batch_size'],uniform_over_groups=False,grouper=grouper,n_groups_per_batch=2)
    test_loaders, max_test_examples = get_test_loaders(config)
    val_loaders, max_val_examples = get_val_loaders(config)

    # Create model.
    net = build_model(config)

    # Use CUDA if desired.
    logging.info(f'cuda device count: {torch.cuda.device_count()}')
    if config['use_cuda']:
        # Often makes things faster, by benchmarking and figuring out how to optimize.
        cudnn.benchmark = True
        device = torch.device("cuda")
        net.cuda(device)

    logging.info('Using cuda? %d', next(net.parameters()).is_cuda)
    # Loss, optimizer, scheduler.
    # Can use a custom loss that takes in a model, inputs, labels, and gets an array of values.
    # For example if you want to regularize weights in some special way.
    # More commonly, we use a criterion, which takes in model_outputs, labels, and outputs a loss.
    # criterion must be specified anyway, since that's the loss we evaluate on test sets.
    criterion = utils.initialize(config['criterion'])
    optimizer = utils.initialize(
        config['lp_optimizer'], update_args={'params': net.parameters()})
    scheduler = utils.initialize(
        config['lp_scheduler'], update_args={'optimizer': optimizer})
    # Training loop.
    best_stats = {}
    best_accs = {}  # Used to save checkpoints of best models on some datasets.
    train_metrics = []
    val_metrics = []
    test_metrics = []
    prev_ckp_path = None
    weight_dict_initial = None
    lam_item = config['lam_item']
    # ------------------------------------------------------------------------------------------------------
    ## Linear Probing Training (Optional)
    for epoch in range(config['lp_epochs']):
        print('LP')
        cur_ckp_filename = 'ckp_' + str(epoch)
        utils.save_ckp(epoch, net, optimizer, scheduler, checkpoints_dir, cur_ckp_filename)
        prev_ckp_path = checkpoints_dir / cur_ckp_filename

        train_stats = train(
           args, mask_dir, bbox_df, dataset, mask_img_ids, bbox_img_ids, img_ids, epoch, config, train_loader,  net, device, optimizer, criterion,
           weight_dict_initial, lam_item)

        # Get test stats across all test sets.
        val_stats = get_all_test_stats(
            epoch, val_loaders, max_test_examples, config, net, criterion, device,
            log_dir=log_dir, loss_name_prefix='val_loss/', acc_name_prefix='val_acc/',f1_name_prefix='val_f1/')

        test_stats = get_all_test_stats(
            epoch, test_loaders, max_test_examples, config, net, criterion, device,
            log_dir=log_dir, loss_name_prefix='test_loss/', acc_name_prefix='test_acc/',f1_name_prefix='test_f1/')
        # Keep track of the best stats.
        update_best_stats(train_stats, best_stats)
        update_best_stats(val_stats, best_stats)
        # Log and save stats.
        train_metrics.append(train_stats)
        val_metrics.append(val_stats)
        test_metrics.append(test_stats)
        train_df = pd.DataFrame(train_metrics)
        val_df = pd.DataFrame(val_metrics)
        test_df = pd.DataFrame(test_metrics)
        df = train_df.merge(val_df, on='epoch')
        df = df.merge(test_df, on='epoch')
        assert(len(df) == len(train_df) == len(val_df))
        df.to_csv(log_dir + '/stats.tsv', sep='\t')

        utils.save_json(log_dir + '/current_train.json', train_stats)
        utils.save_json(log_dir + '/current_test.json', test_stats)
        utils.save_json(log_dir + '/current_val.json', val_stats)
        utils.save_json(log_dir + '/best.json', best_stats)
        # Save checkpoint of best model. We save the 'best' for each of a list
        # of specified valid datasets. For example, we might want to save the best
        # model according to in-domain validation metrics, but as an oracle, save
        # the best according to ood validation metrics (or a proxy ood metric).
        if 'early_stop_dataset_names' in config:
            logging.info(f"Early stopping using datasets {config['early_stop_dataset_names']}")
            for name in config['early_stop_dataset_names']:
                if name not in val_loaders:
                    raise ValueError(f"{name} is not the name of a test dataset.")
                metric_name = 'val_acc/' + name
                assert(metric_name in val_stats)
                if metric_name not in best_accs or val_stats[metric_name] > best_accs[metric_name]:
                    best_accs[metric_name] = val_stats[metric_name]
                    checkpoint_name = 'ckp_best_' + name

                    if 'save_no_checkpoints' not in config or not config['save_no_checkpoints']:
                        utils.save_ckp(epoch, net, optimizer, scheduler, checkpoints_dir, checkpoint_name)
    # ------------------------------------------------------------------------------------------------------
    ## Full Fine-tuning Training
    for param in net.parameters():
        param.requires_grad = True

    config['batch_size'] = config['batch_size_ft']
    config['linear_probe'] = False

    optimizer = utils.initialize(
        config['ft_optimizer'], update_args={'params': net.parameters()})

    scheduler = utils.initialize(
        config['ft_scheduler'], update_args={'optimizer': optimizer})

    ## Re-define the training loader with config['batch_size'] = config['batch_size_ft']
    train_loader = get_train_loader("standard", train_data, batch_size=config['batch_size'],uniform_over_groups=False,grouper=grouper,n_groups_per_batch=2)

    for epoch in range(config['lp_epochs'], config['lp_epochs']+config['ft_epochs']):

        cur_ckp_filename = 'ckp_' + str(epoch)
        utils.save_ckp(epoch, net, optimizer, scheduler, checkpoints_dir, cur_ckp_filename)
        prev_ckp_path = checkpoints_dir / cur_ckp_filename

        train_stats = train(
          args,mask_dir, bbox_df, dataset, mask_img_ids, bbox_img_ids, img_ids,epoch, config, train_loader, net, device, optimizer, criterion,
           weight_dict_initial, lam_item, is_ft=True)

        val_stats = get_all_test_stats(
            epoch, val_loaders, max_test_examples, config, net, criterion, device,
            log_dir=log_dir, loss_name_prefix='val_loss/', acc_name_prefix='val_acc/',f1_name_prefix='val_f1/')

        test_stats = get_all_test_stats(
            epoch, test_loaders, max_test_examples, config, net, criterion, device,
            log_dir=log_dir, loss_name_prefix='test_loss/', acc_name_prefix='test_acc/',f1_name_prefix='test_f1/')

        # Keep track of the best stats.
        update_best_stats(train_stats, best_stats)
        update_best_stats(val_stats, best_stats)
        # Log and save stats.
        train_metrics.append(train_stats)
        val_metrics.append(val_stats)
        test_metrics.append(test_stats)
        train_df = pd.DataFrame(train_metrics)
        val_df = pd.DataFrame(val_metrics)
        test_df = pd.DataFrame(test_metrics)
        df = train_df.merge(val_df, on='epoch')
        df = df.merge(test_df, on='epoch')
        assert(len(df) == len(train_df) == len(val_df))
        df.to_csv(log_dir + '/stats.tsv', sep='\t')

        utils.save_json(log_dir + '/current_train.json', train_stats)
        utils.save_json(log_dir + '/current_test.json', test_stats)
        utils.save_json(log_dir + '/current_val.json', val_stats)
        utils.save_json(log_dir + '/best.json', best_stats)
        # Save checkpoint of best model. We save the 'best' for each of a list
        # of specified valid datasets. For example, we might want to save the best
        # model according to in-domain validation metrics, but as an oracle, save
        # the best according to ood validation metrics (or a proxy ood metric).
        if 'early_stop_dataset_names' in config:
            logging.info(f"Early stopping using datasets {config['early_stop_dataset_names']}")
            for name in config['early_stop_dataset_names']:
                if name not in val_loaders:
                    raise ValueError(f"{name} is not the name of a test dataset.")
                metric_name = 'val_acc/' + name
                assert(metric_name in val_stats)
                if metric_name not in best_accs or val_stats[metric_name] > best_accs[metric_name]:
                    best_accs[metric_name] = val_stats[metric_name]
                    checkpoint_name = 'ckp_best_' + name
                    if 'save_no_checkpoints' not in config or not config['save_no_checkpoints']:
                        utils.save_ckp(epoch, net, optimizer, scheduler, checkpoints_dir, checkpoint_name)

    # ------------------------------------------------------------------------------------------------------
    ## Continue Full Fine-tuning with Smaller Learning Rate
    config['ft_optimizer']['args']['lr'] = config['ft_optimizer']['args']['lr']*0.1
    optimizer = utils.initialize(
        config['ft_optimizer'], update_args={'params': net.parameters()})

    scheduler = utils.initialize(
        config['ft_scheduler'], update_args={'optimizer': optimizer})

    for epoch in range(config['lp_epochs']+config['ft_epochs'], config['lp_epochs']+config['ft_epochs']+config['ft_continue_epochs']):
        cur_ckp_filename = 'ckp_' + str(epoch)
        utils.save_ckp(epoch, net, optimizer, scheduler, checkpoints_dir, cur_ckp_filename)
        prev_ckp_path = checkpoints_dir / cur_ckp_filename
        train_stats = train(
          args,mask_dir, bbox_df, dataset, mask_img_ids, bbox_img_ids, img_ids,epoch, config, train_loader, net, device, optimizer, criterion,
           weight_dict_initial, lam_item, is_ft=True)

        val_stats = get_all_test_stats(
            epoch, val_loaders, max_test_examples, config, net, criterion, device,
            log_dir=log_dir, loss_name_prefix='val_loss/', acc_name_prefix='val_acc/',f1_name_prefix='val_f1/')

        test_stats = get_all_test_stats(
            epoch, test_loaders, max_test_examples, config, net, criterion, device,
            log_dir=log_dir, loss_name_prefix='test_loss/', acc_name_prefix='test_acc/',f1_name_prefix='test_f1/')
        # Keep track of the best stats.
        update_best_stats(train_stats, best_stats)
        update_best_stats(val_stats, best_stats)
        # Log and save stats.
        train_metrics.append(train_stats)
        val_metrics.append(val_stats)
        test_metrics.append(test_stats)
        train_df = pd.DataFrame(train_metrics)
        val_df = pd.DataFrame(val_metrics)
        test_df = pd.DataFrame(test_metrics)
        df = train_df.merge(val_df, on='epoch')
        df = df.merge(test_df, on='epoch')
        assert(len(df) == len(train_df) == len(val_df))
        df.to_csv(log_dir + '/stats.tsv', sep='\t')

        utils.save_json(log_dir + '/current_train.json', train_stats)
        utils.save_json(log_dir + '/current_test.json', test_stats)
        utils.save_json(log_dir + '/current_val.json', val_stats)
        utils.save_json(log_dir + '/best.json', best_stats)
        # Save checkpoint of best model. We save the 'best' for each of a list
        # of specified valid datasets. For example, we might want to save the best
        # model according to in-domain validation metrics, but as an oracle, save
        # the best according to ood validation metrics (or a proxy ood metric).
        if 'early_stop_dataset_names' in config:
            logging.info(f"Early stopping using datasets {config['early_stop_dataset_names']}")
            for name in config['early_stop_dataset_names']:
                if name not in val_loaders:
                    raise ValueError(f"{name} is not the name of a test dataset.")
                metric_name = 'val_acc/' + name
                assert(metric_name in val_stats)
                if metric_name not in best_accs or val_stats[metric_name] > best_accs[metric_name]:
                    best_accs[metric_name] = val_stats[metric_name]
                    checkpoint_name = 'ckp_best_' + name
                    if 'save_no_checkpoints' not in config or not config['save_no_checkpoints']:
                        utils.save_ckp(epoch, net, optimizer, scheduler, checkpoints_dir, checkpoint_name)
    # ------------------------------------------------------------------------------------------------------
    utils.save_ckp(epoch, net, optimizer, scheduler, checkpoints_dir, 'ckp_last')
    utils.load_ckp(str(checkpoints_dir / checkpoint_name), net)



def setup():
    parser = argparse.ArgumentParser(
        description='Run model')
    parser.add_argument('--config', type=str, metavar='c',
                        help='YAML config', required=True)
    parser.add_argument('--log_dir', type=str, metavar='ld',
                        help='Log directory', required=True)
    parser.add_argument('--root_dir', type=str, metavar='ld',
                        help='Root dir for iWildCam dataset', required=True)
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU to be used.')
    parser.add_argument('--tmp_par_ckp_dir', type=str,
                        help='Temporary directory to save checkpoints instead of log_dir.')
    parser.add_argument('--copy_all_folders', action='store_true',
                        help='Copy all folders (e.g. code, utils) for reproducibility.')
    parser.add_argument('--seed', type=int, default=None, help='random seed')



    args, unparsed = parser.parse_known_args()
    os.environ["HOME"] = os.getcwd()
    # Choose the GPU to use.
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu_id)
    # Make log and checkpoint directories.
    log_dir = args.log_dir
    make_new_dir(log_dir)
    # Sometimes we don't want to overload a distributed file system with checkpoints.
    # So we save checkpoints on a tmp folder on a local machine. Then later we transfer
    # the checkpoints back.
    if args.tmp_par_ckp_dir is not None:
        checkpoints_dir = make_checkpoints_dir(args.tmp_par_ckp_dir)
    else:
        checkpoints_dir = make_checkpoints_dir(log_dir)
    # If you want to copy folders to get the whole state of code
    # while running. For more reproducibility.
    if args.copy_all_folders:
        copy_folders(args.log_dir)
    # Setup logging.
    utils.setup_logging(log_dir, log_level)
    # Open config, update with command line args
    if args.config.endswith('.json'):
        # For json files, we just use it directly and don't process it, e.g. by adding
        # root_dir. Use this for loading saved configurations.
        with open(args.config) as json_file:
            config = json.load(json_file)
    else:
        config = quinine.Quinfig(args.config)
    config['root_dir']=args.root_dir
    # Update config with command line arguments.
    utils.update_config(unparsed, config)
    # This makes specifying certain things more convenient, e.g. don't have to specify a
    # transform for every test datset.
    preprocess_config(config, args.config)
    # If we should save model preds, then save them.
    if 'save_model_preds' in config and config.save_model_preds:
        os.makedirs(log_dir + '/model_preds/')
    # Set seed.
    set_random_seed(config['seed'])
    # Save updated config.
    config_json = log_dir+'/config.json'
    with open(config_json, 'w') as f:
        json.dump(config, f)
    # Save command line arguments.
    save_command_line_args(log_dir)
    return args,config, log_dir, checkpoints_dir, args.tmp_par_ckp_dir


if __name__ == "__main__":
    args,config, log_dir, checkpoints_dir, tmp_par_ckp_dir = setup()
    main(args,config, log_dir, checkpoints_dir)
    if tmp_par_ckp_dir is not None:
        new_checkpoints_dir = log_dir + '/checkpoints'
        logging.info('Copying from %s to %s', checkpoints_dir, new_checkpoints_dir)
        shutil.copytree(checkpoints_dir, new_checkpoints_dir)
