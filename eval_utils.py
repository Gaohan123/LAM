import src.utils.utils as utils
import torch
from sklearn.metrics import f1_score
from src.utils.accumulator import Accumulator
from utils import reset_state
import logging
import numpy as np

def get_test_loaders(config, shuffle=False):
    """
    Load test datasets.
    """
    test_loaders = {}
    max_test_examples = {}
    logging.info('Found %d testing datasets.', len(config['test_datasets']))
    for test_dataset_config in config['test_datasets']:
        logging.info('test dataset config: ' + str(test_dataset_config))
        # Initialize dataset and data loader.
        # Shuffle is True in case we only test part of the test set.
        test_data = utils.init_dataset(test_dataset_config)
        test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=config['batch_size'],
            shuffle=shuffle, num_workers=config['num_workers'])
        test_config_name = test_dataset_config['name']
        test_loaders[test_config_name] = test_loader
        # Some test datasets like CINIC are huge so we only test part of the dataset.
        if 'max_test_examples' in test_dataset_config:
            logging.info(
                'Only logging %d examples for %s', test_dataset_config['max_test_examples'],
                test_dataset_config['name'])
            max_test_examples[test_config_name] = test_dataset_config['max_test_examples']
        else:
            max_test_examples[test_config_name] = float('infinity')
        logging.info('test loader name: ' + test_dataset_config['name'])
        logging.info('test loader: ' + str(test_loader))
        logging.info('test transform: ' + str(test_dataset_config['transforms']))
    return test_loaders, max_test_examples


def get_val_loaders(config, shuffle=False):
    """
    Load validation datasets
    """
    test_loaders = {}
    max_test_examples = {}
    logging.info('Found %d val datasets.', len(config['val_datasets']))
    for test_dataset_config in config['val_datasets']:
        logging.info('val dataset config: ' + str(test_dataset_config))
        # Initialize dataset and data loader.
        # Shuffle is True in case we only test part of the test set.
        test_data = utils.init_dataset(test_dataset_config)
        test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=config['batch_size'],
            shuffle=shuffle, num_workers=config['num_workers'])
        test_config_name = test_dataset_config['name']
        test_loaders[test_config_name] = test_loader
        # Some test datasets like CINIC are huge so we only test part of the dataset.
        if 'max_test_examples' in test_dataset_config:
            logging.info(
                'Only logging %d examples for %s', test_dataset_config['max_test_examples'],
                test_dataset_config['name'])
            max_test_examples[test_config_name] = test_dataset_config['max_test_examples']
        else:
            max_test_examples[test_config_name] = float('infinity')
        logging.info('val loader name: ' + test_dataset_config['name'])
        logging.info('val loader: ' + str(test_loader))
        logging.info('val transform: ' + str(test_dataset_config['transforms']))
    return test_loaders, max_test_examples



def get_test_stats(config, net, test_loader, criterion, device, epoch, loader_name, log_dir,
                   max_examples=float('infinity')):
    # Evaluate accuracy and loss on validation.
    # Returns right after we've seen at least max_examples examples (not batches).
    val_loss = Accumulator()
    val_acc = Accumulator()
    training_state = net.training
    net.eval()
    num_examples = 0
    if 'save_model_preds' in config and config.save_model_preds:
        predicted_list = []
        labels_list = []
    predicted_list1 = []
    labels_list1 = []
    with torch.no_grad():
        for data in test_loader:
            if config['use_cuda']:
                data = utils.to_device(data, device)
            images, labels = data
            images = images[torch.where(labels!=-1)[0]]
            labels = labels[torch.where(labels!=-1)[0]]
            if len(labels) == 0:
                continue
            outputs, cfeatures = net(images)
            if hasattr(test_loader.dataset, 'valid_indices'):
                # This basically projects onto the set of valid indices.
                # We take the argmax among the set of valid indices.
                logits = outputs.data.detach().cpu().numpy()
                predicted = get_argmax_valid(logits, np.arange(logits.shape[1]))
            else:
                _, predicted = torch.max(outputs.data, dim=1)
                predicted = predicted.detach().cpu().numpy()
            if 'save_model_preds' in config and config.save_model_preds:
                predicted_list.append(predicted)
                labels_list.append(labels.detach().cpu().numpy())
            predicted_list1.extend(predicted)
            labels_list1.extend(labels.detach().cpu().numpy())
            correct = (predicted == labels.detach().cpu().numpy())
            val_acc.add_values(correct.tolist())
            y_true=labels.detach().cpu().numpy()
            y_pred=predicted
            loss = criterion(outputs, labels).cpu()
            loss_list = loss.tolist()
            val_loss.add_value(loss_list)
            num_examples += len(images)
            if num_examples >= max_examples:
                logging.info("Breaking after %d examples.", num_examples)
                break
    if 'save_model_preds' in config and config.save_model_preds:
        preds = np.concatenate(predicted_list)
        labels = np.concatenate(labels_list)
        pickle_name = log_dir+'/model_preds/'+loader_name+'_'+str(epoch)+'_preds.pkl'
        pickle.dump((preds, labels), open(pickle_name, "wb"))
    reset_state(net, training_state)
    val_f1=f1_score(labels_list1, predicted_list1, average='macro',labels=np.unique(labels_list1))
    return val_loss, val_acc, val_f1


def get_all_test_stats(epoch, test_loaders, max_test_examples, config, net, criterion, device,
                       log_dir, loss_name_prefix, acc_name_prefix,f1_name_prefix):
    stats = {'epoch': epoch}

    for name, test_loader in test_loaders.items():
        logging.info(f'testing {name}')
        max_examples = float('infinity')
        if name in max_test_examples:
            max_examples = max_test_examples[name]
        val_loss, val_acc,val_f1 = get_test_stats(
            config, net, test_loader, criterion, device, epoch, name,
            log_dir=log_dir, max_examples=max_examples)
        stats[loss_name_prefix + name] = val_loss.get_mean()
        stats[acc_name_prefix + name] = val_acc.get_mean()
        stats[f1_name_prefix + name] = val_f1#.get_mean()
    return stats
