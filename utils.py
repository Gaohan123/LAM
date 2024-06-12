import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import os
import shutil
import logging
from pathlib import Path
import sys


def make_new_dir(new_dir):
    os.makedirs(new_dir,exist_ok=True)

def make_checkpoints_dir(log_dir):
    checkpoints_dir = log_dir + '/checkpoints'
    checkpoints_dir = Path(checkpoints_dir).resolve().expanduser()
    os.makedirs(checkpoints_dir,exist_ok=True)
    return checkpoints_dir

def copy_folders(log_dir):
    copy_folders = ['code', 'configs', 'scripts', 'lib', 'configs', 'models',
                    'experiments', 'utils', 'examples', 'src', 'datasets']
    for copy_folder in copy_folders:
        if os.path.isdir('./' + copy_folder):
            shutil.copytree('./' + copy_folder, log_dir + '/' + copy_folder)

def save_command_line_args(log_dir):
    command = ""
    command += sys.executable + " "
    command += " ".join(sys.argv)
    logging.info('Command: ' + command)
    with open(log_dir+'/command.txt', 'w') as f:
        f.write(command)
        f.write('\n')


def update_train_transform(config):
    if 'no_augmentation' in config and config['no_augmentation']:
        if 'default_test_transforms' not in config:
            raise ValueError('If no_augmentation=True, must specify default_test_transforms.')
        config['train_dataset']['transforms'] = config['default_test_transforms']


def update_test_transform_args_configs(config):
    # Use default test transform for test datasets that don't specify a transform.
    for test_dataset_config in config['test_datasets']:
        if 'transforms' not in test_dataset_config:
            if config['default_test_transforms'] is None:
                raise ValueError('Must either specify default_test_transforms '
                                 'or a transform for each test dataset')
            test_dataset_config['transforms'] = config['default_test_transforms']
        if 'default_test_args' in config:
            if config['default_test_args'] is not None:
                for default_test_arg in config['default_test_args']:
                    if default_test_arg not in test_dataset_config['args']:
                        test_dataset_config['args'][default_test_arg] = config['default_test_args'][default_test_arg]

    for val_dataset_config in config['val_datasets']:
        if 'transforms' not in val_dataset_config:
            if config['default_test_transforms'] is None:
                raise ValueError('Must either specify default_test_transforms '
                                 'or a transform for each test dataset')
            val_dataset_config['transforms'] = config['default_test_transforms']


def update_root_prefix(config):
    # Go through test datasets, and train dataset. If root_prefix specified, then prepend that
    # to the root.
    def apply_root_prefix(dataset_config, root_prefix):
        for key in ['root', 'cache_path', 'pickle_file_path']:
            if key in dataset_config['args']:
                orig_path = dataset_config['args'][key]
                logging.info('orig_path %s', orig_path)
                dataset_config['args'][key] = root_prefix + '/' + orig_path

    if 'root_prefix' in config:
        root_prefix = config['root_prefix']
        logging.info("Adding root prefix %s to all roots.", root_prefix)
        for val_dataset_config in config['val_datasets']:
            apply_root_prefix(val_dataset_config, root_prefix)
        for test_dataset_config in config['test_datasets']:
            apply_root_prefix(test_dataset_config, root_prefix)


def set_random_seed(seed):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed + 111)


def preprocess_config(config, config_path):
    # If it's not a json config (e.g. if it's yaml) then process it.
    if not config_path.endswith('.json'):
        # If we don't specify a transform for some test datasets, but specify a default transform,
        # then use the default transform for that dataset. For datasets that do specify a transform
        # we use that and not the default transform.
        update_test_transform_args_configs(config)
        # Datasets may be stored in different directories in different clusters and platforms.
        # We allow specifying a root_prefix that gets prepended to any specified dataset roots.
        # So if config['root_prefix'] is defined then we prepend it to dataset['args']['root'] for
        # train and test datasets.
        update_root_prefix(config)
        # # Note: copying config over is not that useful anymore with Quinine, so use json below.
        # shutil.copy(args.config, log_dir+'/original_config.yaml')
        # If no_augmentation option in config, then use test_transforms for training.
        update_train_transform(config)


def reset_state(model, training):
    if training:
        model.train()
    else:
        model.eval()

def update_best_stats(stats, best_stats):
    for k, v in stats.items():
        best_k = 'best_' + k
        if best_k in best_stats:
            cmb = max
            if k.find('loss') != -1 and k.find('acc') == -1:
                cmb = min
            best_stats[best_k] = cmb(best_stats[best_k], v)
        else:
            best_stats[best_k] = v

# In some datasets, only some indices are valid.
# So we only take the argmax among the set of valid indices.
def get_argmax_valid(logits, valid_indices):
    compressed_logits = logits[:, valid_indices]
    assert compressed_logits.shape == (logits.shape[0], len(np.unique(valid_indices)))
    compressed_preds = np.argmax(compressed_logits, axis=-1).astype(np.int32)
    valid_indices = np.array(valid_indices)
    preds = valid_indices[compressed_preds]
    assert preds.shape == (logits.shape[0],)
    return preds
