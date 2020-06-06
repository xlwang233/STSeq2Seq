import argparse
import os
import numpy as np
import pandas as pd
import time
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA, ARMA

from utils import utils
from utils.metrics import masked_rmse_np, masked_mape_np, masked_mae_np
from sklearn.svm import SVR

import warnings
from statsmodels.tools.sm_exceptions import HessianInversionWarning, ConvergenceWarning
warnings.simplefilter("ignore", HessianInversionWarning)
warnings.simplefilter("ignore", ConvergenceWarning)


def historical_average_predict(df, period=12 * 24 * 7, test_ratio=0.2, null_val=0.):
    """
    Calculates the historical average of sensor reading.
    From: https://github.com/liyaguang/DCRNN
    :param df:
    :param period: default 1 week.
    :param test_ratio:
    :param null_val: default 0.
    :return:
    """
    n_sample, n_sensor = df.shape
    n_test = int(round(n_sample * test_ratio))
    n_train = n_sample - n_test
    y_test = df[-n_test:]
    y_predict = pd.DataFrame.copy(y_test)

    for i in range(n_train, min(n_sample, n_train + period)):
        inds = [j for j in range(i % period, n_train, period)]
        historical = df.iloc[inds, :]
        y_predict.iloc[i - n_train, :] = historical[historical != null_val].mean()
    # Copy each period.
    for i in range(n_train + period, n_sample, period):
        size = min(period, n_sample - i)
        start = i - n_train
        y_predict.iloc[start:start + size, :] = y_predict.iloc[start - period: start + size - period, :].values
    return y_predict, y_test


def arima_predict(df, horizons):
    road_list = df.columns.values.tolist()  # get the list of roads
    n_sample, n_roads = df.shape
    n_test = 6850  # for METR-LA
    # n_test = 10419  # for PEMS-BAY
    n_train = n_sample - n_test
    # df_train, df_test = df[:n_train], df[n_train:]

    max_horizon = max(horizons)

    # get the time series for every single road.
    ts = []
    for road_id in road_list:
        ts.append(df[road_id])

    res = np.zeros(shape=(len(horizons), n_test, n_roads))  # (4, 6850, 207)
    start_ind = n_train - max_horizon  # 34272 - 6850 - 12 = 27410
    # end_ind = n_sample - max_horizon  # 34272 - 12 = 34260

    for k, series in enumerate(ts):
        # for ind in range(start_ind, end_ind):
        print("start process {0}th road...".format(k))
        start_time = time.time()
        for i in range(n_test):  # 0 ~ 6849
            try:
                st_ind = start_ind + i + 1 - (12 * 6)
                train_dat = series[st_ind:start_ind+i+1].values  # use past 6 hours data
                # order = sm.tsa.arma_order_select_ic(train_dat, max_ar=12, max_ma=2, ic='aic')['aic_min_order']
                arma_model = ARMA(train_dat, order=(3, 1))
                arma_result = arma_model.fit(disp=0)
                prediction = arma_result.forecast(steps=max_horizon)[0]  # (12, )
                for h, horizon in enumerate(horizons):
                    res[h, i, k] = prediction[horizon-1]
            except:
                pass
            continue
        end_time = time.time()
        print("time elapsed: {:.4f}s".format(end_time-start_time))
    return res


def svr_predict(x_train, y_train, x_test, horizons):
    """
    :param x_train: (n_train_samples, his_len=12, n_roads)
    :param y_train: (n_train_samples, horizon=12, n_roads)
    :param x_test: (n_test_samples, his_len=12, n_roads)
    :param horizons: target horizons, defaults to (1, 3, 6, 12)
    :return: y_predicts: (n_horizons, n_test_samples, n_roads)
    """
    n_train, his_len, n_roads = x_train.shape
    n_test, _, _ = x_test.shape
    svr = SVR(C=0.1, cache_size=200, coef0=1.0, degree=3, epsilon=0.001, gamma='auto',
              kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
    y_predicts = np.zeros((len(horizons), n_test, n_roads))
    for i, h in enumerate(horizons):
        start_time = time.time()
        for r in range(n_roads):
            X_train_dat = x_train[..., r]  # (n_train, his_len)
            y_train_dat = y_train[:, h-1, r]  # (n_train,)
            X_test_dat = x_test[..., r]  # (n_test, his_len)
            svr.fit(X_train_dat, y_train_dat)
            y_test = svr.predict(X_test_dat)  # (n_test,) ndarray
            y_predicts[i, :, r] = y_test
        end_time = time.time()
        compute_time_info = "computation time for {0}th horizon: {1:.4f}s".format(h, end_time-start_time)
        print(compute_time_info)
        logger.info(compute_time_info)
    return y_predicts


def eval_historical_average(traffic_reading_df, period):
    y_predict, y_test = historical_average_predict(traffic_reading_df, period=period, test_ratio=0.2)
    rmse = masked_rmse_np(preds=y_predict.values, labels=y_test.values, null_val=0)
    mape = masked_mape_np(preds=y_predict.as_matrix(), labels=y_test.as_matrix(), null_val=0)
    mae = masked_mae_np(preds=y_predict.as_matrix(), labels=y_test.as_matrix(), null_val=0)
    logger.info('Historical Average')
    logger.info('\t'.join(['Model', 'Horizon', 'RMSE', 'MAPE', 'MAE']))
    for horizon in [1, 3, 6, 12]:
        line = 'HA\t%d\t%.2f\t%.2f\t%.2f' % (horizon, rmse, mape * 100, mae)
        logger.info(line)


def eval_arima(traffic_reading_df, y_test):
    horizons = [1, 3, 6, 12]
    y_predicts = arima_predict(traffic_reading_df, horizons=horizons)
    # y_predicts: (len(horizons), n_test, n_roads)  y_test: (12, n_test, n_roads)
    # outputs = {
    #     'predictions': y_predicts,
    #     'groundtruth': y_test
    # }
    # np.savez_compressed('saved/baselines/arima_6h.npz', **outputs)

    logger.info('ARIMA  pdq: (3, 0, 1)')
    logger.info('Model\tHorizon\tRMSE\tMAPE\tMAE')
    for i, horizon in enumerate(horizons):
        rmse = masked_rmse_np(preds=y_predicts[i], labels=y_test[horizon-1], null_val=0)
        mape = masked_mape_np(preds=y_predicts[i], labels=y_test[horizon-1], null_val=0)
        mae = masked_mae_np(preds=y_predicts[i], labels=y_test[horizon-1], null_val=0)
        line = 'ARIMA\t%d\t%.2f\t%.2f\t%.2f' % (horizon, rmse, mape * 100, mae)
        logger.info(line)


def eval_svr(dataset_dir):
    # load data
    data = {}
    for category in ['train', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        data['x_' + category] = cat_data['x'][..., 0]  # (n_samples, seq_len, n_roads)
        data['y_' + category] = cat_data['y'][..., 0]  # (n_samples, horizon=12, n_roads)
    scaler = utils.StandardScaler(mean=data['x_train'].mean(), std=data['x_train'].std())
    # Data normalization
    for category in ['train', 'test']:
        data['x_' + category] = scaler.transform(data['x_' + category])
        if category == "train":
            data['y_' + category] = scaler.transform(data['y_' + category])

    horizons = [3, 6, 12]
    y_predicts = svr_predict(data['x_train'], data['y_train'], data['x_test'], horizons=horizons)
    # y_predicts: (len(horizons), n_test, n_roads)  y_test: (n_test, horizon, n_roads)
    y_predicts = scaler.inverse_transform(y_predicts)

    logger.info('SVR.  number of past observations: 12')
    logger.info('Model\t Horizon\t RMSE\t MAPE\t MAE')
    for i, horizon in enumerate(horizons):
        rmse = masked_rmse_np(preds=y_predicts[i], labels=data['y_test'][:, horizon - 1, :], null_val=0)
        mape = masked_mape_np(preds=y_predicts[i], labels=data['y_test'][:, horizon - 1, :], null_val=0)
        mae = masked_mae_np(preds=y_predicts[i], labels=data['y_test'][:, horizon - 1, :], null_val=0)
        line = 'SVR\t %d\t %.2f\t %.2f\t %.2f' % (horizon, rmse, mape * 100, mae)
        logger.info(line)


def main(args):
    # load data
    traffic_reading_df = pd.read_hdf(args.traffic_reading_filename)

    # eval HA
    start_time = time.time()
    eval_historical_average(traffic_reading_df, period=7 * 24 * 12)
    end_time = time.time()
    time_log = "HA time elapsed: {:.4f}s".format(end_time - start_time)
    print(time_log)
    logger.info(time_log)

    # # eval ARIMA.
    # Right now the implementation is not feasible because it takes too much time to run.
    # test_data = np.load("../data/METR-LA/test.npz")
    # test_y = test_data['y']  # (6850, 12, 207, 2)
    # test_y = test_y[..., 0]  # (6850, 12, 207)
    # test_y = np.transpose(test_y, (1, 0, 2))  # (12, 6850, 207)
    # start_time = time.time()
    # eval_arima(traffic_reading_df, y_test=test_y)
    # end_time = time.time()
    # time_log = "ARIMA time elapsed: {:.4f}s".format(end_time - start_time)
    # print(time_log)
    # logger.info(time_log)

    # eval SVR
    start_time = time.time()
    eval_svr(args.dataset_dir)
    end_time = time.time()
    time_log = "SVR time elapsed: {:.4f}s".format(end_time-start_time)
    print(time_log)
    logger.info(time_log)


if __name__ == '__main__':
    logger = utils.get_logger('../saved/baselines/METR-LA', 'Baseline')
    parser = argparse.ArgumentParser()
    parser.add_argument('--traffic_reading_filename', default="../data/METR-LA/metr-la.h5", type=str,
                        help='Path to the traffic Dataframe.')
    parser.add_argument('--dataset_dir', default='../data/METR-LA', type=str,
                        help="Path to train/val/test dataset.")
    args = parser.parse_args()
    logger.info("dataset dir: {}".format(args.dataset_dir))
    main(args)
