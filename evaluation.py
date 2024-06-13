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

import ssl
ssl._create_default_https_context = ssl._create_unverified_context
log_level = logging.INFO




def main(args,config, log_dir, checkpoints_dir):
    """
    Main function
    """
    # Set up datasets and loaders.
    print("Entering main.")
    logging.info("Entering main.")

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
    criterion = utils.initialize(config['criterion'])
    val_metrics = []
    test_metrics = []
    # ------------------------------------------------------------------------------------------------------
    for param in net.parameters():
        param.requires_grad = False

    # Get test stats across all test sets.
    val_stats = get_all_test_stats(
        0, val_loaders, max_test_examples, config, net, criterion, device,
        log_dir=log_dir, loss_name_prefix='val_loss/', acc_name_prefix='val_acc/',f1_name_prefix='val_f1/')

    test_stats = get_all_test_stats(
        0, test_loaders, max_test_examples, config, net, criterion, device,
        log_dir=log_dir, loss_name_prefix='test_loss/', acc_name_prefix='test_acc/',f1_name_prefix='test_f1/')

    # Log and save stats.
    val_metrics.append(val_stats)
    test_metrics.append(test_stats)
    val_df = pd.DataFrame(val_metrics)
    test_df = pd.DataFrame(test_metrics)
    df = val_df.merge(val_df, on='epoch')
    df = df.merge(test_df, on='epoch')
    df.to_csv(log_dir + '/stats.tsv', sep='\t')




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
