import numpy as np

np.random.seed(123)  # for reproducibility
import pandas as pd
import requests
import os
import pickle as pk
import configparser
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


def save_pkl(pkl_file, file_path, file_name, min_v=None, max_v=None):
    pk.dump(pkl_file, open(os.path.join(file_path, file_name), "wb"))
    print(file_name, ' is dumped in: ', file_path)
    if ((min_v is not None) and (max_v is not None)):
        pk.dump(min_v, open(os.path.join(file_path, file_name + "min"), "wb"))
        print(file_name + ".min", ' is dumped in: ', file_path)
        pk.dump(max_v, open(os.path.join(file_path, file_name + "max"), "wb"))
        print(file_name + ".max", ' is dumped in: ', file_path)


def load_pkl(file_path, file_name):
    pkl_file = pk.load(open(os.path.join(file_path, file_name), "rb"))
    print(file_name, ' is loaded from: ', file_path)
    return pkl_file


def split_data(data, labels, ratio_=0.8, shuffle=False):
    if shuffle:
        data_size = len(data)
        shuffle_indices = np.random.permutation(np.arange(data_size))
        data = data[shuffle_indices]
        labels = labels[shuffle_indices]

    train_X = data[:int(len(data) * ratio_)]
    test_X = data[int(len(data) * ratio_):]

    train_Y = labels[:int(len(labels) * ratio_)]
    test_Y = labels[int(len(labels) * ratio_):]

    return train_X, train_Y, test_X, test_Y


def get_ndarray_by_sliding_window(data_df, input_len, output_len, vars_list, only_target=False):
    i = 0
    X = []
    Y = []
    while True:
        if (i + input_len + output_len) <= len(data_df):
            X.append(data_df[i:i + input_len][vars_list].values)
            Y.append(data_df[i + input_len:i + input_len + output_len][vars_list].values)
            i += 1
        else:
            X = np.array(X)
            Y = np.array(Y)
            assert len(Y) == len(X), 'Length Error !!!!!!!'
            break
    return X, Y


def get_train_test(data_df, input_len, output_len, var_name, per=0.9, only_target=True, data_name='obs'):
    i = 0
    X = []
    Y = []
    # if data_name=='obs':
    #    targets = ['t2m_obs','rh2m_obs','w10m_obs','psur_obs', 'q2m_obs', 'd10m_obs', 'u10m_obs', 'v10m_obs', 'RAIN_obs']

    # elif data_name=='ruitu':

    #    targets = ['psfc_M', 't2m_M', 'q2m_M', 'rh2m_M', 'w10m_M', 'd10m_M', 'u10m_M',
    #   'v10m_M', 'SWD_M', 'GLW_M', 'HFX_M', 'LH_M', 'RAIN_M', 'PBLH_M',
    #   'TC975_M', 'TC925_M', 'TC850_M', 'TC700_M', 'TC500_M', 'wspd975_M',
    #   'wspd925_M', 'wspd850_M', 'wspd700_M', 'wspd500_M', 'Q975_M', 'Q925_M',
    #   'Q850_M', 'Q700_M', 'Q500_M']
    targets = var_name[data_name]

    while True:
        if (i + input_len + output_len) <= len(data_df):
            X.append(data_df[i:i + input_len][targets].values)
            Y.append(data_df[i + input_len:i + input_len + output_len][targets].values)
            i += 1
        else:
            X = np.array(X)
            Y = np.array(Y)
            assert len(Y) == len(X), 'Length Error'
            break

    train_X, train_Y, test_X, test_Y = split_data(X, Y, ratio_=per)

    if data_name == 'ruitu':
        return train_X, train_Y, test_X, test_Y

    elif (data_name == 'obs' and only_target):
        # Only return the first three variables, i.e., ['t2m_obs','rh2m_obs','w10m_obs']
        return train_X[:, :, :3], train_Y[:, :, :3], test_X[:, :, :3], test_Y[:, :, :3]

    else:
        return train_X, train_Y[:, :, :3], test_X, test_Y[:, :, :3]


def rmse(y_pred, y_true):
    y_pred = y_pred.reshape(-1)
    y_true = y_true.reshape(-1)
    # print(y_true.shape)
    return np.sqrt(mean_squared_error(y_pred, y_true))


def mse(y_pred, y_true):
    y_pred = y_pred.reshape(-1)
    y_true = y_true.reshape(-1)
    return mean_squared_error(y_pred, y_true)


def bias(y_pred, y_true):
    pass


def mae(y_pred, y_true):
    y_pred = y_pred.reshape(-1)
    y_true = y_true.reshape(-1)
    return np.mean(np.abs(y_pred - y_true))


def score(y_pred, y_true):
    pass


def evl_fn(y_pred, y_true, **kwargs):
    renorm = kwargs['renorm']
    min_v = kwargs['max_min'][0]
    max_v = kwargs['max_min'][1]

    if renorm:
        y_pred = y_pred * (max_v - min_v) + min_v
        y_true = y_true * (max_v - min_v) + min_v

        print('\t rmse:', rmse(y_pred, y_true))
        print('\t mae: ', mae(y_pred, y_true))
        print('\t mse: ', mse(y_pred, y_true))
    else:
        print('\t rmse:', rmse(y_pred, y_true))
        print('\t mae: ', mae(y_pred, y_true))
        print('\t mse: ', mse(y_pred, y_true))


# print('Baseline_direct: rmse:{}, mse:{}'.format(rmse(X,Y), mse(X, Y)))

def renorm(norm_value, min_v, max_v):
    real_v = norm_value * (max_v - min_v) + min_v
    return real_v


def cal_miss(X_miss):
    nums_ = len(X_miss.reshape(-1))
    miss_nums = np.sum(X_miss == -9999)
    print('all nums:', nums_)
    print('missing nums:', miss_nums)
    print('missing ratio:', miss_nums / nums_)


def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.
    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]


def intplt_nan_1d(data_nan, obs_data_nan, sta_id):  # Default stationID
    '''
    data_nan: is Ruitu data with np.NaN
    sta_id: Is only one stationID;
    obs_data_nan: is Observation data with np.NaN

    '''
    data_nan[data_nan == -9999] = np.NaN
    obs_data_nan[obs_data_nan == -9999] = np.NaN

    data_nan = data_nan[:, :, sta_id]
    obs_data_nan = obs_data_nan[:, :, sta_id]

    new_list = []
    print('Original Ruitu Data Shape:', data_nan.shape)
    print('Original Observed Data Shape:', obs_data_nan.shape)

    # print('Firstly, we delete the totally lost days in Obs dataset and the counterpart day in Ruitu dataset.')
    day_should_deleted = []
    for i in range(obs_data_nan.shape[0]):
        if np.isnan(obs_data_nan[i, :]).any():
            if sum(np.isnan(obs_data_nan[i, :])) == 37:
                day_should_deleted.append(i)
                continue
    print('Data are totally lost during the days in obs dataset!', day_should_deleted)
    obs_data_nan = np.array(np.delete(obs_data_nan, day_should_deleted, 0))
    data_nan = np.array(np.delete(data_nan, day_should_deleted, 0))
    # ---------------------------------------------------------

    # print('Secondly, we delete the totally lost days in Ruitu dataset and the counterpart day in Obs dataset.')
    day_should_deleted = []
    for i in range(data_nan.shape[0]):
        if np.isnan(data_nan[i, :]).any():
            if sum(np.isnan(data_nan[i, :])) == 37:
                day_should_deleted.append(i)
                continue
    print('Data are totally lost during the days in Ruitu dataset!', day_should_deleted)
    obs_data_nan = np.array(np.delete(obs_data_nan, day_should_deleted, 0))
    data_nan = np.array(np.delete(data_nan, day_should_deleted, 0))
    # ---------------------------------------------------------

    ### Interpolate for Input data
    for i in range(data_nan.shape[0]):
        # print(i)
        new_X = data_nan[i, :].copy()
        if np.isnan(new_X).any():
            nans, x_temp = nan_helper(new_X)
            new_X[nans] = np.interp(x_temp(nans), x_temp(~nans), new_X[~nans])
        new_list.append(new_X)
    data_after_intplt = np.array(new_list)

    ###Interpolate for Label(Obs) data
    Y_list = []
    for i in range(obs_data_nan.shape[0]):
        new_Y = obs_data_nan[i, :].copy()
        if np.isnan(new_Y).any():
            # print('Miss happen! Interpolate...')
            nans, y_temp = nan_helper(new_Y)
            # print(np.isnan(new_Y))
            new_Y[nans] = np.interp(y_temp(nans), y_temp(~nans), new_Y[~nans])
        Y_list.append(new_Y)

    obs_after_intplt = np.array(Y_list)
    print('After interpolate, Ruitu Data Shape:', data_after_intplt.shape)
    print('After interpolate, Observed Data Shape:', obs_after_intplt.shape)
    return data_after_intplt, obs_after_intplt


def min_max_norm(org_data, min_, max_):
    return (org_data - min_) / (max_ - min_)
