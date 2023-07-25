import pickle as pk
import numpy as np
import os
import scipy.sparse as sp
import torch
from scipy.sparse import linalg
from torch.autograd import Variable
import h5py




def normal_std(x):
    return x.std() * np.sqrt((len(x) - 1.)/(len(x)))

def load_pkl(file_path, file_name):
    pkl_file = pk.load(open(os.path.join(file_path, file_name), "rb"))
    print(file_name, ' is loaded from: ', file_path)
    return pkl_file

class DataLoaderS(object):
    # train and valid is the ratio of training set and validation set. test = 1 - train - valid
    def __init__(self, file_name, train, valid, device, horizon, window, normalize=2):
        self.P = window
        self.h = horizon
        fin = open(file_name)
        self.rawdat = np.loadtxt(fin, delimiter=',')
        self.dat = np.zeros(self.rawdat.shape)
        self.n, self.m = self.dat.shape
        self.normalize = 2
        self.scale = np.ones(self.m)
        self._normalized(normalize)
        self._split(int(train * self.n), int((train + valid) * self.n), self.n)

        self.scale = torch.from_numpy(self.scale).float()
        tmp = self.test[1] * self.scale.expand(self.test[1].size(0), self.m)

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

        train_set = range(self.P + self.h - 1, train)
        valid_set = range(train, valid)
        test_set = range(valid, self.n)
        self.train = self._batchify(train_set, self.h)
        self.valid = self._batchify(valid_set, self.h)
        self.test = self._batchify(test_set, self.h)

    def _batchify(self, idx_set, horizon):
        n = len(idx_set)
        X = torch.zeros((n, self.P, self.m))
        Y = torch.zeros((n, self.m))
        for i in range(n):
            end = idx_set[i] - self.h + 1
            start = end - self.P
            X[i, :, :] = torch.from_numpy(self.dat[start:end, :])
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

class DataLoaderM(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True):
        """
        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
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
                yield (x_i, y_i)
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
    def inverse_transform(self, data):
        return (data * self.std) + self.mean

class MaxMinScaler():
    """
    Standard the input
    """
    def __init__(self, max, min):
        self.max = max
        self.min = min
    def transform(self, data):
        return (data - self.min) / self.max
    def inverse_transform(self, data):
        return (data * self.max) + self.min


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

class MaxMin():
    """
    max-min scale
    """
    def __init__(self, max, min, GPU="cuda:1"):
        self.device = torch.device(GPU)
        self.max = max
        self.min = min

    def transform(self, data):
        return (data - self.min) / (self.max - self.min)
    
    def inverse_transform(self, data, idx=None):
        if not torch.is_tensor(self.max):
            self.max = self.max.reshape((1, -1))
            self.min = self.min.reshape((1, -1))
            self.max = torch.from_numpy(self.max[:, :data.shape[2]])
            self.max = self.max.to(self.device)
            self.min = torch.from_numpy(self.min[:, :data.shape[2]])
            self.min = self.min.to(self.device)

        data = data.transpose(2, 3) #
        out = data * (self.max.float()-self.min.float()) + self.min.float()
        return out.transpose(2, 3)
    
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


def load_dataset(dataset_dir, batch_size, valid_batch_size= None, test_batch_size=None):
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

    file = h5py.File(dataset_dir+'weather2k_dict.h5', "r")
    group = file["my_dict"]

    data = {}
    for category in ['train', 'val', 'test']:
        tmp = group['x_' + category][()]
        # build time feat
        # 1. switch axis to fit former's format
        data['x_' + category] = tmp.transpose((0, 3, 1, 2))[:,:,:200,3:4] # temperature
        
        tmp_y = group['y_' + category][()]
        tmp_y = tmp_y.transpose((0, 3, 1, 2))[:,:,:200,2:3]
        data['y_' + category] = np.expand_dims(tmp_y, axis=-1) # [samples, time, station, 3, 1]

    scaler = MaxMinScaler(max=48, min=-18)

    # data standardization
    for category in ['train', 'val', 'test']:
        data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])
        
    data['train_loader'] = DataLoaderM(data['x_train'], data['y_train'], batch_size)
    data['val_loader'] = DataLoaderM(data['x_val'], data['y_val'], batch_size)
    data['test_loader'] = DataLoaderM(data['x_test'], data['y_test'], batch_size)
    data['scaler'] = scaler

    return data


def load_dataset_all(dataset_dir, batch_size, valid_batch_size= None, test_batch_size=None, data_name='BJ'):
    '''
    
    '''
    if data_name == 'Israel':
        return new_load_Israel_dataset(dataset_dir, batch_size, valid_batch_size=None, test_batch_size=None)
    elif data_name == 'USA':
        return load_USA_dataset(dataset_dir, batch_size, valid_batch_size=None, test_batch_size=None)
    else:
        
        data = {}
        # choose one meteorological variable
        # 0 for t2m, 1 for rh2m, 2 for w10m
        var_idx = 0
        obs_range = [[-30,42], [0.0,100.0], [0.0, 30.0]]
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
            data['x_' + category] = np.concatenate(tmp_list, axis=-1)[:, :-9, :, :, :] # [n, time, station, variables, features] feats(hour, value)
            # only the first three variables as label same with WB's KDD
            data['y_' + category] = np.expand_dims(cat_data['ground_truth'][:, 4:, :, :], axis=-1) # [samples, 37, 10, 3, 1]
            
            
            data['x_' + category] = data['x_' + category][:, :, :, var_idx, :]
            data['y_' + category] = data['y_' + category][:, :, :, var_idx, :]
            
        # standardization    
        scaler = MaxMinScaler(max=obs_range[var_idx][1], min=obs_range[var_idx][0])

        # data standardization
        for category in ['train', 'val', 'test']:
            data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])

        data['train_loader'] = DataLoaderM(data['x_train'], data['y_train'], batch_size)
        data['val_loader'] = DataLoaderM(data['x_val'], data['y_val'], valid_batch_size)
        data['test_loader'] = DataLoaderM(data['x_test'], data['y_test'], test_batch_size)
        data['scaler'] = scaler

        return data


def load_USA_dataset(dataset_dir, batch_size, valid_batch_size= None, test_batch_size=None):
    '''
    load data, include: data
    data includes [n_sample, stations, variable, features]
    '''
    # data['x_train'] data['y_train']
    ori_data = load_pkl(dataset_dir, 'USA_data.dict')
    
    USA_range = [(-35.0, 40.0), (0.0, 40.0), (800.0, 1100.0), (0.0, 100.0)]

    # choose one meteorological variable
    # 0 for t2m, 1 for rh2m, 2 for w10m
    var_idx = 0
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
    
        data['x_' + category] = data['x_' + category][:, :, :, var_idx, :]
        data['y_' + category] = data['y_' + category][:, :, :, var_idx, :]
 
    scaler = MaxMinScaler(max=USA_range[var_idx][1], min=USA_range[var_idx][0])

    # data standardization
    for category in ['train', 'val', 'test']:
        data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])

    data['train_loader'] = DataLoaderM(data['x_train'], data['y_train'], batch_size)
    data['val_loader'] = DataLoaderM(data['x_val'], data['y_val'], batch_size)
    data['test_loader'] = DataLoaderM(data['x_test'], data['y_test'], batch_size)
    data['scaler'] = scaler

    return data


def load_Israel_dataset(dataset_dir, batch_size, valid_batch_size= None, test_batch_size=None):
    '''
    load data, include: data
    data includes [n_sample, stations, variable, features]
    '''
    # data['x_train'] data['y_train']
    Israel_range = [(-20.0, 50.0), (0.0, 60.0), (900.0, 1100.0), (0.0, 100.0)]
    
    var_idx=0
    
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

        data['x_' + category] = data['x_' + category][:, :, :, var_idx, :]
        data['y_' + category] = data['y_' + category][:, :, :, var_idx, :]
 
    scaler = MaxMinScaler(max=USA_range[var_idx][1], min=USA_range[var_idx][0])

    # data standardization
    for category in ['train', 'val', 'test']:
        data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])

    data['train_loader'] = DataLoaderM(data['x_train'], data['y_train'], batch_size)
    data['val_loader'] = DataLoaderM(data['x_val'], data['y_val'], batch_size)
    data['test_loader'] = DataLoaderM(data['x_test'], data['y_test'], batch_size)
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


def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
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
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)/labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def metric(pred, real):
    mae = masked_mae(pred,real,0.0).item()
    mape = masked_mape(pred,real,0.0).item()
    rmse = masked_rmse(pred,real,0.0).item()
    return mae,mape,rmse


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



            