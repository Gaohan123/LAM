
import torch
import torch.nn
from ..utils import utils as utils
import logging

def set_linear_layer(layer, coef, intercept):
    coef_tensor = torch.tensor(coef, dtype=layer.weight.dtype).cuda()
    bias_tensor = torch.tensor(intercept, dtype=layer.bias.dtype).cuda()
    coef_param = torch.nn.parameter.Parameter(coef_tensor)
    bias_param = torch.nn.parameter.Parameter(bias_tensor)
    layer.weight = coef_param
    layer.bias = bias_param

def build_model(config):
    """
    Initialize pretrained model
    """
    net = utils.initialize(config['model'])

    def count_parameters(model, trainable):
        return sum(p.numel() for p in model.parameters() if p.requires_grad == trainable)

    logging.info('linear probing, freezing bottom layers.')
    # If unspecified, we set use_net_val_mode = True for linear-probing.
    # We did this in update_net_eval_mode which we called in main.
    # assert('use_net_val_mode' in config)
    # Freeze all the existing weights of the neural network.
    net.set_requires_grad(False)

    if 'probe_net' in config:
        probe_net = utils.initialize(config['probe_net'])
        net.add_probe(probe_net)
    else:
        net.new_last_layer(config['train_dataset']['num_classes'])

    num_trainable_params = count_parameters(net, True)
    num_params = count_parameters(net, False) + num_trainable_params
    logging.info(f'Fine Tuning {num_trainable_params} of {num_params} parameters.')

    if 'checkpoint_path' in config and len(config['checkpoint_path']) > 0:
        logging.info(utils.load_ckp(config['checkpoint_path'], net))
        num_trainable_params = count_parameters(net, True)
        num_params = count_parameters(net, False) + num_trainable_params
        logging.info(f'Fine Tuning checkpoint: {num_trainable_params} of {num_params} parameters.')
    return net


def get_param_weights_counts(net, detach):
    weight_dict = {}
    count_dict = {}
    for param in net.named_parameters():
        name = param[0]
        weights = param[1]
        if detach:
            weight_dict[name] = weights.detach().clone()
        else:
            weight_dict[name] = weights
        count_dict[name] = np.prod(np.array(list(param[1].shape)))
    return weight_dict, count_dict
