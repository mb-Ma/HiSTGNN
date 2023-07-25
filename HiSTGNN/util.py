import pickle
import numpy as np
import os
import scipy.sparse as sp
import torch
from scipy.sparse import linalg
from torch.autograd import Variable
from helper import *
from process_meteo import obs_range_dic, ruitu_range_dic
from process_Israel import Israel_range
from process_USA import USA_range
import sys
import h5py


def normal_std(x):
    return x.std() * np.sqrt((len(x) - 1.)/(len(x)))


class DataLoaderS(object):
    # train and valid is the ratio of training set and validation set. test = 1 - train - valid
    def __init__(self, file_name, train, valid, device, horizon, window, normalize=2):
        self.P = window # period
        self.h = horizon # 3
        fin = open(file_name)
        self.rawdat = np.loadtxt(fin, delimiter=',')

        self.dat = np.zeros(self.rawdat.shape) # 原始数据的维度  构造一个二维矩阵
        self.n, self.m = self.dat.shape # n 时间维度， m 站点个数
        self.normalize = 2 # 归一化处理的操作ID
        self.scale = np.ones(self.m) # 构建一个m样本特征的1向量  scale [1, m]
        self._normalized(normalize) # 按除以列最大值进行标准化， scale的值表示每一列最大的绝对值
        self._split(int(train * self.n), int((train + valid) * self.n), self.n) # 0.6*n  0.8*n  n  return 训练 测试 验证

        self.scale = torch.from_numpy(self.scale).float()
        # test_Y was scaled, why do this operation again?
        # ans: reduction original value
        # ask: how to reduction? do not storage max value
        tmp = self.test[1] * self.scale.expand(self.test[1].size(0), self.m) # self.test[1]: test_Y self.test[0]: test_X

        self.scale = self.scale.to(device)
        self.scale = Variable(self.scale)

        self.rse = normal_std(tmp)
        self.rae = torch.mean(torch.abs(tmp - torch.mean(tmp)))

        self.device = device

    def _normalized(self, normalize):
        # normalized by the maximum value of entire matrix.

        if (normalize == 0):
            self.dat = self.rawdat

        if (normalize == 1):
            self.dat = self.rawdat / np.max(self.rawdat)

        # normlized by the maximum value of each row(sensor).
        if (normalize == 2):
            for i in range(self.m):
                self.scale[i] = np.max(np.abs(self.rawdat[:, i]))
                self.dat[:, i] = self.rawdat[:, i] / np.max(np.abs(self.rawdat[:, i]))

    def _split(self, train, valid, test):

        train_set = range(self.P + self.h - 1, train) # 7*24+3-1, 0.6*n  (p+h-1, n)这么多个样本
        valid_set = range(train, valid) # 0.6*n 0.8*n
        test_set = range(valid, self.n) # 0.8*n n
        self.train = self._batchify(train_set, self.h)
        self.valid = self._batchify(valid_set, self.h)
        self.test = self._batchify(test_set, self.h)

    def _batchify(self, idx_set, horizon):
        n = len(idx_set) # 样本长度
        X = torch.zeros((n, self.P, self.m)) # 样本长度 窗口长度 站点数
        Y = torch.zeros((n, self.m)) # 样本长度 特征个数
        for i in range(n):
            end = idx_set[i] - self.h + 1
            start = end - self.P
            X[i, :, :] = torch.from_numpy(self.dat[start:end, :]) # self.dat was scaled
            Y[i, :] = torch.from_numpy(self.dat[idx_set[i], :])
        return [X, Y]

    def get_batches(self, inputs, targets, batch_size, shuffle=True):
        length = len(inputs)
        if shuffle:
            index = torch.randperm(length)
        else:
            index = torch.LongTensor(range(length))
        start_idx = 0
        while (start_idx < length):
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            X = inputs[excerpt]
            Y = targets[excerpt]
            X = X.to(self.device)
            Y = Y.to(self.device)
            yield Variable(X), Variable(Y)
            start_idx += batch_size


class DataLoaderM_A(object):
    def __init__(self, xs, ys, lg, gg, batch_size, pad_with_last_sample=True):
        """
        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0

        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size  # 求补足的个数
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            lg_padding = np.repeat(lg[-1:], num_padding, axis=0)
            gg_padding = np.repeat(gg[-1:], num_padding, axis=0)

            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
            lg = np.concatenate([lg, lg_padding], axis=0)
            gg = np.concatenate([gg, gg_padding], axis=0)

        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys
        self.lg = lg
        self.gg = gg

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        lg, gg = self.lg[permutation], self.gg[permutation]
        self.xs = xs
        self.ys = ys
        self.lg = lg
        self.gg = gg

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                lg_i = self.lg[start_ind: end_ind, ...]
                gg_i = self.gg[start_ind: end_ind, ...]

                yield x_i, y_i, lg_i, gg_i
                self.current_ind += 1

        return _wrapper()


class DataLoaderM(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True, is_windows=False):
        """
        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0

        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size # 求补足的个数
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                yield x_i, y_i
                self.current_ind += 1

        return _wrapper()


class StandardScaler():
    """
    Standard the input
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def y_transform(self, data):
        return (data - self.mean[:3]) / self.std[:3]

    def inverse_transform(self, data):
        return (data * self.std) + self.mean

    def inverse_transform2(self, data, idx=None):
        device = torch.device("cuda:0") # self.min, self.max = self.min[idx], self.max[idx]
        if not torch.is_tensor(self.mean):
            self.mean = self.mean.reshape((1, -1))
            self.std = self.std.reshape((1, -1))
            self.mean = torch.from_numpy(self.mean)
            self.mean = self.mean.to(device)
            self.std = torch.from_numpy(self.std)
            self.std = self.std.to(device)

        data = data.transpose(1, 3).transpose(2, 3) # [batch, output_num, 1, features]
        out = data * self.std.float() + self.mean.float()
        return out.transpose(2, 3).transpose(1, 3)


class MaxMin():
    """
    max-min scale
    """
    def __init__(self, max, min, GPU="cuda:0"):
        self.device = torch.device(GPU)
        self.max = max
        self.min = min

    def transform(self, data):
        return (data - self.min) / (self.max - self.min)

    def y_transform(self, data):
        return (data - self.min[:3]) / (self.max[:3] - self.min[:3])

    def transform_svds(self, data):
        var1 = [i for i in range(data.shape[0]) if i % 3 == 0]
        var2 = [i for i in range(data.shape[0]) if i % 3 == 1]
        var3 = [i for i in range(data.shape[0]) if i % 3 == 2]
        data[var1, :, :] = (data[var1, :, :] - self.min[0]) / (self.max[0] - self.min[0])
        data[var2, :, :] = (data[var2, :, :] - self.min[1]) / (self.max[1] - self.min[1])
        data[var3, :, :] = (data[var3, :, :] - self.min[2]) / (self.max[2] - self.min[2])
        return data

    def inverse_transform(self, data, idx=None):
        if not torch.is_tensor(self.max):
            self.max = self.max.reshape((1, -1))
            self.min = self.min.reshape((1, -1))
            self.max = torch.from_numpy(self.max)
            self.max = self.max.to(self.device)
            self.min = torch.from_numpy(self.min)
            self.min = self.min.to(self.device)
        if not torch.is_tensor(idx): # 如果没有打乱，这里应该是None,如果有打乱的数值，应该为Tensor
            idx = [i for i in range(self.min.shape[1])]
            idx = torch.tensor(idx).to(self.device)
        self.min = self.min[:, idx]
        self.max = self.max[:, idx]

        data = data.transpose(1, 3).transpose(2, 3) # [batch, output_num, 1, features]
        out = data * (self.max.float()-self.min.float()) + self.min.float()
        return out.transpose(2, 3).transpose(1, 3)

    def inverse_transform2(self, data, idx=None):
        if not torch.is_tensor(self.max):
            self.max = self.max.reshape((1, -1))
            self.min = self.min.reshape((1, -1))
            self.max = torch.from_numpy(self.max[:, :data.shape[3]])
            self.max = self.max.to(self.device)
            self.min = torch.from_numpy(self.min[:, :data.shape[3]])
            self.min = self.min.to(self.device)

        data = data.transpose(3, 4) #
        out = data * (self.max.float()-self.min.float()) + self.min.float()
        return out.transpose(3, 4)
    
    def inverse_transform3(self, data, idx=None):
        '''
        for weather2k
        '''
        if not torch.is_tensor(self.max):
            self.max = self.max.reshape((1, -1))
            self.min = self.min.reshape((1, -1))
            self.max = torch.from_numpy(self.max[:, [1,2,3]])
            self.max = self.max.to(self.device)
            self.min = torch.from_numpy(self.min[:, [1,2,3]])
            self.min = self.min.to(self.device)

        data = data.transpose(3, 4) #
        out = data * (self.max.float()-self.min.float()) + self.min.float()
        return out.transpose(3, 4)
 

def sym_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()


def asym_adj(adj):
    """Asymmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat= sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()


def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian


def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32).todense()


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


def load_adj(pkl_filename):
    sensor_ids, sensor_id_to_ind, adj = load_pickle(pkl_filename)
    return adj


def ori_load_dataset(dataset_dir, batch_size, valid_batch_size= None, test_batch_size=None, scale='std'):
    data = {}
    # add time of day feature
    for category in ['train', 'val', 'test']:
        cat_data = load_pkl(dataset_dir, category + '.dict')
        # only one station and add time-of-day feature
        tmp = cat_data['input_obs'][:, :-9, 0, :]
        num_samples, num_windows, num_nodes = tmp.shape
        tmp = np.expand_dims(tmp, axis=-1)
        tmp_list = [tmp]
        time_ind = [((i + 3) % 24) / 24 for i in range(num_windows)]  # hour in day
        time_ind = np.array(time_ind)
        time_in_day = np.tile(time_ind, [num_samples, 1, num_nodes, 1]).transpose((0, 3, 2, 1))
        tmp_list.append(time_in_day)
        data['x_' + category] = np.concatenate(tmp_list, axis=-1)

        # only one station for label
        data['y_' + category] = np.expand_dims(cat_data['ground_truth'][:, :, 0, :], axis=-1)

    if scale == 'std':
        scaler = StandardScaler(mean=data['x_train'][..., 0].reshape((-1, 9)).mean(axis=0),
                                std=data['x_train'][..., 0].reshape((-1, 9)).std(axis=0))
    else:
        scaler = MaxMin(max=np.array([v[1] for k, v in obs_range_dic.items()]).reshape(1, 9), min=np.array([v[0] for k, v in obs_range_dic.items()]).reshape(1, 9))

    # data standardization
    for category in ['train', 'val', 'test']:
        data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])

    data['train_loader'] = DataLoaderM(data['x_train'], data['y_train'], batch_size, is_windows=True)
    data['val_loader'] = DataLoaderM(data['x_val'], data['y_val'], valid_batch_size, is_windows=True)
    data['test_loader'] = DataLoaderM(data['x_test'], data['y_test'], test_batch_size, is_windows=True)
    data['scaler'] = scaler

    return data


def load_dataset(dataset_dir, batch_size, valid_batch_size= None, test_batch_size=None, scale='std'):
    '''
    load data, include: train_X, train_Y, test valid...
    '''
    data = {}
    for category in ['train', 'val', 'test']:
        cat_data = load_pkl(dataset_dir, category + '.dict')
        # build time feature
        tmp = cat_data['input_obs'][:, :, :, :] # [n, time, stations, variables]
        num_samples, num_windows, num_station, num_variables = tmp.shape
        tmp = np.expand_dims(tmp, axis=-1)
        tmp_list = [tmp]
        time_ind = [((i + 3) % 24) / 24 for i in range(num_windows)]  # hour in day
        time_ind = np.array(time_ind) # 37
        time_in_day = np.tile(time_ind, [num_samples, 1, num_station, num_variables, 1]).transpose((0, 4, 2, 3, 1))
        tmp_list.append(time_in_day)
        data['x_' + category] = np.concatenate(tmp_list, axis=-1) # [n, time, station, variables, features] feats(hour, station, value)
        # only the first three variables as label same with WB's KDD
        data['y_' + category] = np.expand_dims(cat_data['ground_truth'][:, :, :, :3], axis=-1) # [samples, 37, 10, 3, 1]

        # wanna put all station in samples
        data['x_' + category] = data['x_' + category].transpose((0, 2, 1, 3, 4))
        data['x_' + category] = data['x_' + category].reshape((-1, num_windows, num_variables, 2), order='C')
        data['y_' + category] = data['y_' + category].transpose((0, 2, 1, 3, 4))
        data['y_' + category] = data['y_' + category].reshape((-1, num_windows, 3, 1), order='C')

        # add station feature for X data
        num_samples = data['x_' + category].shape[0]
        stat_feat = np.array([i % num_station for i in range(num_samples)])
        stat_feat = np.tile(stat_feat, [1, num_windows, num_variables, 1]).transpose((3, 1, 2, 0))
        data['x_' + category] = np.concatenate([data['x_' + category], stat_feat], axis=-1)


    # standardization
    if scale == 'std':
        scaler = StandardScaler(mean=np.array([data['x_train'][:, :, i, 0].mean() for i in range(9)]),
                                std=np.array([data['x_train'][:, :, i, 0].mean() for i in range(9)]))
    else: #max-min standardization
        scaler = MaxMin(max=np.array([v[1] for k, v in obs_range_dic.items()]), min=np.array([v[0] for k, v in obs_range_dic.items()]))

    # data standardization
    for category in ['train', 'val', 'test']:
        data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])

    data['train_loader'] = DataLoaderM(data['x_train'], data['y_train'], batch_size)
    data['val_loader'] = DataLoaderM(data['x_val'], data['y_val'], valid_batch_size)
    data['test_loader'] = DataLoaderM(data['x_test'], data['y_test'], test_batch_size)
    data['scaler'] = scaler

    return data


def load_dataset_all(dataset_dir, batch_size, valid_batch_size= None, test_batch_size=None, scale='std', predA=True, data_name='WFD', num_var=9, GPU="cuda:0"):
    '''
    load data, include: train_X, train_Y, test valid...
    data includes [n_sample, stations, variable, features]
    '''
    if data_name == 'Israel':
        return new_load_Israel_dataset(dataset_dir, batch_size, valid_batch_size=None, test_batch_size=None, scale=scale,
                                   predA=True, GPU=GPU)
    elif data_name == 'USA':
        return load_USA_dataset(dataset_dir, batch_size, valid_batch_size=None, test_batch_size=None, scale=scale,
                                   predA=True, GPU=GPU)
    elif data_name == "2k":
        return load_2k_dataset(dataset_dir, batch_size, valid_batch_size=None, test_batch_size=None, scale=scale,
                                   predA=True, GPU=GPU)
    else:
        data = {}
        for category in ['train', 'val', 'test']:
            cat_data = load_pkl(dataset_dir, category + '.dict')
            # build time feature
            tmp = cat_data['input_obs'][:, :, :, :] # [n, time, stations, variables]
            num_samples, num_windows, num_station, num_variables = tmp.shape
            tmp = np.expand_dims(tmp, axis=-1)
            tmp_list = [tmp]
            time_ind = [((i + 3) % 24) / 24 for i in range(num_windows)]  # hour in day
            time_ind = np.array(time_ind) # 37
            time_in_day = np.tile(time_ind, [num_samples, 1, num_station, num_variables, 1]).transpose((0, 4, 2, 3, 1))
            tmp_list.append(time_in_day)
            data['x_' + category] = np.concatenate(tmp_list, axis=-1)[:, :-9, :, :num_var, :] # [n, time, station, variables, features] feats(hour, value)
            # only the first three variables as label same with WB's KDD
            data['y_' + category] = np.expand_dims(cat_data['ground_truth'][:, 4:, :, :1], axis=-1) # [samples, 37, 10, 3, 1]

        # standardization
        if scale == 'std':
            scaler = StandardScaler(mean=np.array([data['x_train'][:, :, :, i, 0].mean() for i in range(num_var)]),
                                    std=np.array([data['x_train' ][:, :, :, i, 0].mean() for i in range(num_var)]))
        else: #max-min standardization
            scaler = MaxMin(max=np.array([v[1] for k, v in obs_range_dic.items()])[:num_var], min=np.array([v[0] for k, v in obs_range_dic.items()])[:num_var], GPU=GPU)

        # data standardization
        for category in ['train', 'val', 'test']:
            data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])

        data['train_loader'] = DataLoaderM(data['x_train'], data['y_train'], batch_size)
        data['val_loader'] = DataLoaderM(data['x_val'], data['y_val'], valid_batch_size)
        data['test_loader'] = DataLoaderM(data['x_test'], data['y_test'], test_batch_size)
        data['scaler'] = scaler

        return data


def load_USA_dataset(dataset_dir, batch_size, valid_batch_size= None, test_batch_size=None, scale='std', predA=True, GPU="cuda:0"):
    '''
    load data, include: data
    data includes [n_sample, stations, variable, features]
    '''
    # data['x_train'] data['y_train']
    ori_data = load_pkl(dataset_dir, 'USA_data.dict')
    data = {}
    for category in ['train', 'val', 'test']:
        tmp = ori_data['x_' + category]
        # build time feat
        # 1. switch axis to fit former's format
        tmp = tmp.transpose((0, 1, 3, 2))
        num_samples, num_windows, num_station, num_variables = tmp.shape
        tmp = np.expand_dims(tmp, axis=-1)
        tmp_list = [tmp]
        time_ind = [(i % 24) / 24 for i in range(num_windows)]  # hour in day
        time_ind = np.array(time_ind) # 48
        time_in_day = np.tile(time_ind, [num_samples, 1, num_station, num_variables, 1]).transpose((0, 4, 2, 3, 1))
        tmp_list.append(time_in_day)
        data['x_' + category] = np.concatenate(tmp_list, axis=-1) # [n, time, station, variables, features] feats(hour, value)
        tmp_y = ori_data['y_' + category]
        tmp_y = tmp_y.transpose((0, 1, 3, 2))
        data['y_' + category] = np.expand_dims(tmp_y, axis=-1) # [samples, time, station, 3, 1]

    # standardization
    if scale == 'std':
        scaler = StandardScaler(mean=np.array([data['x_train'][:, :, :, i, 0].mean() for i in range(num_variables)]),
                                std=np.array([data['x_train'][:, :, :, i, 0].mean() for i in range(num_variables)]))
    else: # max-min standardization
        scaler = MaxMin(max=np.array([v for k, v in USA_range]), min=np.array([k for k, v in USA_range]), GPU=GPU)

    # data standardization
    for category in ['train', 'val', 'test']:
        data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])

    data['train_loader'] = DataLoaderM(data['x_train'], data['y_train'], batch_size)
    data['val_loader'] = DataLoaderM(data['x_val'], data['y_val'], batch_size)
    data['test_loader'] = DataLoaderM(data['x_test'], data['y_test'], batch_size)
    data['scaler'] = scaler

    return data


def load_Israel_dataset(dataset_dir, batch_size, valid_batch_size= None, test_batch_size=None, scale='std', predA=True, GPU="cuda:0"):
    '''
    load data, include: data
    data includes [n_sample, stations, variable, features]
    '''
    # data['x_train'] data['y_train']
    ori_data = load_pkl(dataset_dir, 'Israel_data.dict')
    data = {}
    for category in ['train', 'val', 'test']:
        tmp = ori_data['x_' + category]
        # build time feat
        # 1. switch axis to fit former's format
        tmp = tmp.transpose((0, 1, 3, 2))
        num_samples, num_windows, num_station, num_variables = tmp.shape
        tmp = np.expand_dims(tmp, axis=-1)
        tmp_list = [tmp]
        time_ind = [(i % 24) / 24 for i in range(num_windows)]  # hour in day
        time_ind = np.array(time_ind) # 48
        time_in_day = np.tile(time_ind, [num_samples, 1, num_station, num_variables, 1]).transpose((0, 4, 2, 3, 1))
        tmp_list.append(time_in_day)
        data['x_' + category] = np.concatenate(tmp_list, axis=-1) # [n, time, station, variables, features] feats(hour, value)
        tmp_y = ori_data['y_' + category]
        tmp_y = tmp_y.transpose((0, 1, 3, 2))
        data['y_' + category] = np.expand_dims(tmp_y, axis=-1) # [samples, time, station, 3, 1]

    # standardization
    if scale == 'std':
        scaler = StandardScaler(mean=np.array([data['x_train'][:, :, :, i, 0].mean() for i in range(num_variables)]),
                                std=np.array([data['x_train'][:, :, :, i, 0].mean() for i in range(num_variables)]))
    else: # max-min standardization
        scaler = MaxMin(max=np.array([v for k, v in Israel_range]), min=np.array([k for k, v in Israel_range]), GPU=GPU)

    # data standardization
    for category in ['train', 'val', 'test']:
        data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])

    data['train_loader'] = DataLoaderM(data['x_train'], data['y_train'], batch_size)
    data['val_loader'] = DataLoaderM(data['x_val'], data['y_val'], batch_size)
    data['test_loader'] = DataLoaderM(data['x_test'], data['y_test'], batch_size)
    data['scaler'] = scaler

    return data


def new_load_Israel_dataset(dataset_dir, batch_size, valid_batch_size= None, test_batch_size=None, scale='std', predA=True, GPU="cuda:0"):
    '''
    load data, include: data
    data includes [n_sample, stations, variable, features]
    '''
    # data['x_train'] data['y_train']
    ori_data = load_pkl(dataset_dir, 'new_Israel_data.dict')
    data = {}
    for category in ['train', 'val', 'test']:
        tmp = ori_data['x_' + category]
        # build time feat
        # 1. switch axis to fit former's format
        # import pdb; pdb.set_trace()
        tmp = tmp.transpose((0, 1, 3, 2))
        num_samples, num_windows, num_station, num_variables = tmp.shape
        data['x_'+category] = np.expand_dims(tmp, axis=-1)
        
        tmp_y = ori_data['y_' + category]
        tmp_y = tmp_y.transpose((0, 1, 3, 2))
        data['y_' + category] = np.expand_dims(tmp_y, axis=-1) # [samples, time, station, 3, 1]

    # standardization
    if scale == 'std':
        scaler = StandardScaler(mean=np.array([data['x_train'][:, :, :, i, 0].mean() for i in range(num_variables)]),
                                std=np.array([data['x_train'][:, :, :, i, 0].mean() for i in range(num_variables)]))
    else: # max-min standardization
        scaler = MaxMin(max=np.array([v for k, v in Israel_range]), min=np.array([k for k, v in Israel_range]), GPU=GPU)

    # data standardization
    for category in ['train', 'val', 'test']:
        data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])

    data['train_loader'] = DataLoaderM(data['x_train'], data['y_train'], batch_size)
    data['val_loader'] = DataLoaderM(data['x_val'], data['y_val'], batch_size)
    data['test_loader'] = DataLoaderM(data['x_test'], data['y_test'], batch_size)
    data['scaler'] = scaler

    return data


def load_2k_dataset(dataset_dir, batch_size, valid_batch_size= None, test_batch_size=None, scale='std', predA=True, GPU="cuda:0"):
    '''
    load data, include: data
    data includes [n_sample,  stations, variable, features]
    '''
    k_range = [(560, 1050), (-18, 48), (0, 100), (0, 59), (-21, 55), (0, 30000), (0, 30000), (0, 30000)]
    k_range_dict = {
        'pressure': [560, 1050],
        'temperature': [-18, 48],
        'rh': [0, 100],
        'wind_speed': [0, 59],
        'land_temperature': [-21, 55],
        'hv1': [0, 30000],
        'hv10': [0, 30000],
        'vv': [0, 30000]
    }

    file = h5py.File(dataset_dir, "r")
    group = file["my_dict"]

    data = {}
    for category in ['train', 'val', 'test']:
        tmp = group['x_' + category][()]
        # build time feat
        # 1. switch axis to fit former's format
        tmp = tmp.transpose((0, 3, 1, 2))[:,:,:200,:]
        data['x_' + category] = np.expand_dims(tmp, axis=-1) # [samples, time, station, 8, 1]
        
        tmp_y = group['y_' + category][()]
        tmp_y = tmp_y.transpose((0, 3, 1, 2))[:,:,:200,:]
        data['y_' + category] = np.expand_dims(tmp_y, axis=-1) # [samples, time, station, 3, 1]

    # standardization
    if scale == 'std':
        scaler = StandardScaler(mean=np.array([data['x_train'][:, :, :, i, 0].mean() for i in range(num_variables)]),
                                std=np.array([data['x_train'][:, :, :, i, 0].mean() for i in range(num_variables)]))
    else: # max-min standardization
        scaler = MaxMin(max=np.array([v for k, v in k_range]), min=np.array([k for k, v in k_range]), GPU=GPU)

    # data standardization
    for category in ['train', 'val', 'test']:
        data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])
    # import pdb; pdb.set_trace()
    data['train_loader'] = DataLoaderM(data['x_train'], data['y_train'], batch_size)
    data['val_loader'] = DataLoaderM(data['x_val'], data['y_val'], batch_size)
    data['test_loader'] = DataLoaderM(data['x_test'], data['y_test'], batch_size)
    data['scaler'] = scaler

    return data


def load_dataset_attribute(dataset_dir, batch_size, valid_batch_size= None, test_batch_size=None, scale='std'):
    '''
    load data, include: train_X, train_Y, test valid...
    '''
    data = {}
    for category in ['train', 'val', 'test']:
        cat_data = load_pkl(dataset_dir, category + '.dict')
        # build time feature
        data['x_' + category] = cat_data['input_obs'] #
        data['y_' + category] = cat_data['ground_truth'][:, :, :, :3]

    # feature standardization
    if scale == 'std':
        scaler = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())
    else: # max-min standardization
        scaler = MaxMin(max=np.array([v[1] for k, v in obs_range_dic.items()]), min=np.array([v[0] for k, v in obs_range_dic.items()]))

    for category in ['train', 'val', 'test']:
        data['x_' + category] = scaler.transform(data['x_' + category])

    data['train_loader'] = DataLoaderM(data['x_train'], data['y_train'], batch_size)
    data['val_loader'] = DataLoaderM(data['x_val'], data['y_val'], valid_batch_size)
    data['test_loader'] = DataLoaderM(data['x_test'], data['y_test'], test_batch_size)
    data['scaler'] = scaler

    return data


def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds-labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_rmse_mean(preds, labels, null_val=np.nan):
    rmse1 = torch.sqrt(masked_mse(preds=preds[:, :, 0, :], labels=labels[:, :, 0, :], null_val=null_val))
    rmse2 = torch.sqrt(masked_mse(preds=preds[:, :, 1, :], labels=labels[:, :, 1, :], null_val=null_val))
    rmse3 = torch.sqrt(masked_mse(preds=preds[:, :, 2, :], labels=labels[:, :, 2, :], null_val=null_val))

    return (rmse1 + rmse2 + rmse3) / 3.0


def masked_stat_rmse_mean(preds, labels, null_val=np.nan):
    rmse1 = torch.sqrt(masked_mse(preds=preds[:, :, :, 0, :], labels=labels[:, :, :, 0, :], null_val=null_val))
    rmse2 = torch.sqrt(masked_mse(preds=preds[:, :, :, 1, :], labels=labels[:, :, :, 1, :], null_val=null_val))
    rmse3 = torch.sqrt(masked_mse(preds=preds[:, :, :, 2, :], labels=labels[:, :, :, 2, :], null_val=null_val))

    return (rmse1 + rmse2 + rmse3) / 3.0


def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def my_loss(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)/labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    # loss = torch.nan_to_num(loss)
    return torch.mean(loss)


def metric_avg(pred, real, single=False):
    '''
    the flag of single indicates if 3d (single-variable) or 4d(multi-variable) data dimensionality.

    '''
    mae = []
    mape = []
    rmse = []
    if not single:
        for i in range(pred.shape[3]):
            mae.append(masked_mae(pred[:, :, :, i, :], real[:, :, :, i, :]).item())
            rmse.append(masked_rmse(pred[:, :, :, i, :], real[:, :, :, i, :]).item())
            mape.append(masked_mape(pred[:, :, :, i, :], real[:, :, :, i, :], null_val=0.0).item())
    else:
        for i in range(pred.shape[3]):
            mae.append(masked_mae(pred[:, :, i, :], real[:, :, i, :]).item())
            rmse.append(masked_rmse(pred[:, :, i, :], real[:, :, i, :]).item())
            mape.append(masked_mape(pred[:, :, i, :], real[:, :, i, :], null_val=0.0).item())
    return np.mean(mae), np.mean(mape), np.mean(rmse)


def metric(pred, real, single_flag=False, single_var=False):
    mae = masked_mae(pred,real, 0.0).item()
    mape = masked_mape(pred,real, 0.0).item()
    if single_flag:
        rmse = masked_rmse_mean(pred, real, 0.0).item()
    elif single_var:
        rmse = masked_rmse(pred, real, 0.0).item()
    else:
        rmse = masked_stat_rmse_mean(pred, real, 0.0).item()

    return mae,mape,rmse


def metrics_ori(pred, real):
    mae = masked_mae(pred, real).item()
    mape = masked_mape(pred, real).item()
    rmse = masked_rmse(pred, real, 0.0).item()
    return mae, mape, rmse


def load_node_feature(path):
    fi = open(path)
    x = []
    for li in fi:
        li = li.strip()
        li = li.split(",")
        e = [float(t) for t in li[1:]]
        x.append(e)
    x = np.array(x)
    mean = np.mean(x,axis=0)
    std = np.std(x,axis=0)
    z = torch.tensor((x-mean)/std,dtype=torch.float)
    return z


def normal_std(x):
    return x.std() * np.sqrt((len(x) - 1.) / (len(x)))
