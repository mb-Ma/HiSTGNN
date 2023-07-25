import numpy as np
import pickle as pk
import os
import torch


obs_range_dic={'t2m_obs':[-30,42], # Official value: [-20,42]
                'rh2m_obs':[0.0,100.0],
                'w10m_obs':[0.0, 30.0],
                'psur_obs':[850,1100],
                'q2m_obs':[0,30],
                 'd10m_obs':[0.0, 360.0],
                 'u10m_obs':[-25.0, 20.0], # Official value: [-20,20]
                 'v10m_obs':[-20.0, 20.0],
                 'RAIN_obs':[0.0, 300.0],}


def load_pkl(file_path, file_name):
    pkl_file = pk.load(open(os.path.join(file_path, file_name), "rb"))
    print(file_name, ' is loaded from: ', file_path)
    return pkl_file


class MaxMin():
    """
    max-min scale
    """
    def __init__(self, max, min):
        self.max = max
        self.min = min

    def transform(self, data):
        return (data - self.min) / (self.max - self.min)

    def inverse_transform(self, data):
        device = torch.device("cuda:0")  # self.min, self.max = self.min[idx], self.max[idx]
        if not torch.is_tensor(self.max):
            self.max = self.max.reshape((1, -1))
            self.min = self.min.reshape((1, -1))
            self.max = torch.from_numpy(self.max[:, :4])
            self.max = self.max.to(device)
            self.min = torch.from_numpy(self.min[:, :4])
            self.min = self.min.to(device)

        data = data.transpose(2, 3)  # [batch, output_num, 1, nodes]
        out = data * (self.max.float()-self.min.float()) + self.min.float()
        return out.transpose(2, 3)

    def inverse_transform2(self, data):
        device = torch.device("cuda:0")  # self.min, self.max = self.min[idx], self.max[idx]
        if not torch.is_tensor(self.max):
            self.max = self.max.reshape((1, -1))
            self.min = self.min.reshape((1, -1))
            self.max = torch.from_numpy(self.max)
            self.max = self.max.to(device)
            self.min = torch.from_numpy(self.min)
            self.min = self.min.to(device)

        data = data.transpose(2, 3)  # [batch, output_num, 1, nodes]
        out = data * (self.max.float()-self.min.float()) + self.min.float()
        return out.transpose(2, 3)


def load_dataset(dataset_dir, scale='std', name='BJ'):
    '''
    load data, include: train_X, train_Y, test valid...
    '''
    data = {}
    if name == 'BJ':
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
            data['y_' + category] = np.expand_dims(cat_data['ground_truth'][:, :, :, :], axis=-1) # [samples, 37, 10, 3, 1]

            # wanna put all station in samples
            data['x_' + category] = data['x_' + category].transpose((0, 2, 1, 3, 4))
            data['x_' + category] = data['x_' + category].reshape((-1, num_windows, num_variables, 2), order='C')
            data['y_' + category] = data['y_' + category].transpose((0, 2, 1, 3, 4))
            data['y_' + category] = data['y_' + category].reshape((-1, num_windows, num_variables, 1), order='C')

        # standardization
        scaler = MaxMin(max=np.array([v[1] for k, v in obs_range_dic.items()]), min=np.array([v[0] for k, v in obs_range_dic.items()]))

        for category in ['train', 'val', 'test']:
            data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])
            data['x_' + category] = data['x_' + category][:, :-9, :, :]
            data['y_' + category] = data['y_' + category][:, 4:, :, :]

        # todo 1.padding 2.batch
        return data, scaler
    if name == 'Israel':
        Israel_range = [(-20.0, 50.0), (0.0, 60.0), (900.0, 1100.0), (0.0, 100.0)]
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
            time_ind = np.array(time_ind)  # 48
            time_in_day = np.tile(time_ind, [num_samples, 1, num_station, num_variables, 1]).transpose((0, 4, 2, 3, 1))
            tmp_list.append(time_in_day)
            data['x_' + category] = np.concatenate(tmp_list,
                                                   axis=-1)  # [n, time, station, variables, features] feats(hour, value)
            tmp_y = ori_data['y_' + category]
            tmp_y = tmp_y.transpose((0, 1, 3, 2))
            data['y_' + category] = np.expand_dims(tmp_y, axis=-1)  # [samples, time, station, 3, 1]

            data['x_' + category] = data['x_' + category].transpose((0, 2, 1, 3, 4))
            data['x_' + category] = data['x_' + category].reshape((-1, num_windows, num_variables, 2), order='C')
            data['y_' + category] = data['y_' + category].transpose((0, 2, 1, 3, 4))
            data['y_' + category] = data['y_' + category].reshape((-1, 24, num_variables, 1), order='C')

        scaler = MaxMin(max=np.array([v for k, v in Israel_range]), min=np.array([k for k, v in Israel_range]))

        for category in ['train', 'val', 'test']:
            data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])

        return data, scaler
    if name == 'USA':
        USA_range = [(-35.0, 40.0), (0.0, 40.0), (800.0, 1100.0), (0.0, 100.0)]
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
            time_ind = np.array(time_ind)  # 48
            time_in_day = np.tile(time_ind, [num_samples, 1, num_station, num_variables, 1]).transpose((0, 4, 2, 3, 1))
            tmp_list.append(time_in_day)
            data['x_' + category] = np.concatenate(tmp_list,
                                                   axis=-1)  # [n, time, station, variables, features] feats(hour, value)
            tmp_y = ori_data['y_' + category]
            tmp_y = tmp_y.transpose((0, 1, 3, 2))
            data['y_' + category] = np.expand_dims(tmp_y, axis=-1)  # [samples, time, station, 3, 1]

            data['x_' + category] = data['x_' + category].transpose((0, 2, 1, 3, 4))
            data['x_' + category] = data['x_' + category].reshape((-1, num_windows, num_variables, 2), order='C')
            data['y_' + category] = data['y_' + category].transpose((0, 2, 1, 3, 4))
            data['y_' + category] = data['y_' + category].reshape((-1, 24, num_variables, 1), order='C')

        scaler = MaxMin(max=np.array([v for k, v in USA_range]), min=np.array([k for k, v in USA_range]))

        for category in ['train', 'val', 'test']:
            data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])
        return data, scaler


if __name__ == '__main__':
    '''
    scaler 可以用于归一化和反归一化
    '''
    dataset_dir = 'data/meteo/'
    # data {'x_train':, 'y_train':, 'x_val':, 'y_val':, 'x_test':, 'y_test':}
    # x_train_shape:[sample, times_1, nodes_1, features]
    # y_train_shape:[sample, times_2, nodes_2, 1]
    data, scaler = load_dataset(dataset_dir)

    # todo 输出维度中节点数与输入维度节点数不同，
    # 假设输出的预测维度为 y_hat [batch, times, nodes, 1]
    # y_hat = scaler.inverse_transform(y_hat)
