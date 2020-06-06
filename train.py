import argparse
import collections
import torch

import utils.metrics as module_metric
import model.models as module_arch
from utils.config_parser import ConfigParser
from trainer.trainer import Trainer
from utils import utils
import math


def main(config):
    logger = config.get_logger('train')

    # load speed data and pre-defined graph data
    _, _, adj_mat = utils.load_graph_data(config["dataset"]["graph_file"])
    data = utils.load_dataset(dataset_dir=config["dataset"]["dataset_dir"],
                              batch_size=config["arch"]["args"]["batch_size"],
                              test_batch_size=config["arch"]["args"]["batch_size"])

    for k, v in data.items():
        if hasattr(v, 'shape'):
            print((k, v.shape))

    train_data_loader = data['train_loader']
    val_data_loader = data['val_loader']

    num_train_sample = data['x_train'].shape[0]
    num_val_sample = data['x_val'].shape[0]

    # get number of iterations per epoch for progress bar
    num_train_iteration_per_epoch = math.ceil(num_train_sample / config["arch"]["args"]["batch_size"])
    num_val_iteration_per_epoch = math.ceil(num_val_sample / config["arch"]["args"]["batch_size"])

    # get device
    device = torch.device(config["device"])

    # build model architecture, then print to console
    model = config.initialize('arch', module_arch)
    logger.info(model)
    model.to(device)

    # log number of trainable parameters
    logger.info("number of trainable parameters: {:d}".format(utils.count_parameters(model)))

    # get function handles of loss and metrics
    loss = config.initialize('loss', module_metric, **{"scaler": data['scaler']})
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.initialize('optimizer', torch.optim, trainable_params)
    lr_scheduler = config.initialize('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    # determine model type
    if config["arch"]["type"] == "FNN":
        model_type = "fnn"
        supports = None
    elif config["arch"]["type"] == "GRUSeq2Seq":
        model_type = "seq2seq"
        supports = None
    else:
        model_type = "seq2seq"
        # build adjacency matrices
        filter_type = config['arch']["args"]['filter_type']
        support_list = []
        supports = []
        if filter_type == "dual_random_walk":
            support_list.append(utils.calculate_random_walk_matrix(adj_mat))
            support_list.append(utils.calculate_random_walk_matrix(adj_mat.T))
        elif filter_type == "identity":
            support_list.append(utils.get_identity_mat(num_nodes=int(config['arch']["args"]['num_nodes'])))
        else:
            raise ValueError("Unknown filter type...")
        if filter_type != "None":
            for support in support_list:
                supports.append(utils.build_sparse_matrix(support).to(device))  # to PyTorch sparse tensor

    trainer = Trainer(device, model_type, model, loss, metrics, optimizer,
                      scaler=data["scaler"], config=config,
                      data_loader=train_data_loader,
                      valid_data_loader=val_data_loader,
                      lr_scheduler=lr_scheduler,
                      len_epoch=num_train_iteration_per_epoch,
                      val_len_epoch=num_val_iteration_per_epoch, supports=supports)
    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='STSeq2Seq')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target=('optimizer', 'args', 'lr')),
        CustomArgs(['--bs', '--batch_size'], type=int, target=('data_loader', 'args', 'batch_size'))
    ]
    config = ConfigParser(args, options)
    main(config)
