import argparse
import torch
import numpy as np
import math
import time
import model.models as module_arch
from utils import metrics, utils
from utils.config_parser import ConfigParser
from tqdm import tqdm


def main(config):
    logger = config.get_logger('test')

    # load speed data and pre-defined graph data
    _, _, adj_mat = utils.load_graph_data(config["dataset"]["graph_file"])
    data = utils.load_dataset(dataset_dir=config["dataset"]["dataset_dir"],
                              batch_size=config["arch"]["args"]["batch_size"],
                              test_batch_size=config["arch"]["args"]["batch_size"])

    for k, v in data.items():
        if hasattr(v, 'shape'):
            print((k, v.shape))
    test_data_loader = data['test_loader']
    scaler = data['scaler']
    num_test_iteration = math.ceil(data['x_test'].shape[0] / config["arch"]["args"]["batch_size"])

    # get device
    device = torch.device(config["device"])

    # prepare model for testing
    model = config.initialize('arch', module_arch)
    logger.info(model)
    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

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

    y_preds = torch.FloatTensor([])
    y_truths = data['y_test']
    predictions = []
    groundtruth = list()

    test_start_time = time.time()
    with torch.no_grad():
        for i, (x, y) in tqdm(enumerate(test_data_loader.get_iterator()), total=num_test_iteration):
            x = torch.FloatTensor(x).to(device)
            y = torch.FloatTensor(y).to(device)
            if model_type == "seq2seq":
                outputs = model(supports, x, y, 0)  # (seq_length, batch_size, num_nodes*output_dim)  (12, 50, 207*1)
                y_preds = torch.cat([y_preds, outputs.squeeze().cpu()], dim=1)
            elif model_type == "fnn":
                outputs = model(x)
                outputs = torch.transpose(outputs, 0, 1)
                y_preds = torch.cat([y_preds, outputs.cpu()], dim=1)
    y_preds = torch.transpose(y_preds, 0, 1)
    y_preds = y_preds.detach().numpy()  # cast to numpy array
    inference_time = time.time() - test_start_time
    print("Inference Time: {:.4f}s".format(inference_time))
    print("--------test results--------")
    logger.info("Inference Time: {:.4f}s".format(inference_time))
    mae_ave = []
    mape_ave = []
    rmse_ave = []
    for horizon_i in range(y_truths.shape[1]):
        y_truth = np.squeeze(y_truths[:, horizon_i, :, 0])

        y_pred = scaler.inverse_transform(y_preds[:, horizon_i, :])
        predictions.append(y_pred)
        groundtruth.append(y_truth)

        mae = metrics.masked_mae_np(y_pred[:y_truth.shape[0]], y_truth, null_val=0)
        mae_ave.append(mae)
        mape = metrics.masked_mape_np(y_pred[:y_truth.shape[0]], y_truth, null_val=0)
        mape_ave.append(mape)
        rmse = metrics.masked_rmse_np(y_pred[:y_truth.shape[0]], y_truth, null_val=0)
        rmse_ave.append(rmse)
        print(
            "Horizon {:02d}, MAE: {:.2f}, MAPE: {:.4f}, RMSE: {:.2f}".format(
                horizon_i + 1, mae, mape, rmse
            )
        )
        log = {"Horizon": horizon_i+1, "MAE": mae, "MAPE": mape, "RMSE": rmse}
        logger.info(log)
        # print average metrics
    mae_ave = np.mean(mae_ave)
    mape_ave = np.mean(mape_ave)
    rmse_ave = np.mean(rmse_ave)
    log_ave_metrics = { "Average MAE": mae_ave, "Average MAPE": mape_ave, "Average RMSE": rmse_ave}
    logger.info(log_ave_metrics)

    # serialize test data
    # outputs = {
    #     'predictions': predictions,
    #     'groundtruth': groundtruth
    # }
    # np.savez_compressed('saved/results/METR-LAres.npz', **outputs)
    # print('Predictions saved as {}.'.format('saved/results/METR-LAres.npz'))


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='STSeq2Seq')

    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')

    config = ConfigParser(args)
    main(config)
