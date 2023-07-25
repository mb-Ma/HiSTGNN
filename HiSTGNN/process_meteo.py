import click
import logging
from pathlib import Path
import netCDF4 as nc
import pickle as pk
import pandas as pd
import datetime
import os
import numpy as np
from helper import save_pkl, load_pkl, min_max_norm

obs_var=['t2m_obs','rh2m_obs','w10m_obs','psur_obs', 'q2m_obs', 'd10m_obs', 'u10m_obs', 'v10m_obs', 'RAIN_obs']

target_var=['t2m_obs', 'rh2m_obs', 'w10m_obs']

ruitu_var=['psfc_M', 't2m_M', 'q2m_M', 'rh2m_M', 'w10m_M', 'd10m_M', 'u10m_M',
       'v10m_M', 'SWD_M', 'GLW_M', 'HFX_M', 'LH_M', 'RAIN_M', 'PBLH_M',
       'TC975_M', 'TC925_M', 'TC850_M', 'TC700_M', 'TC500_M', 'wspd975_M',
       'wspd925_M', 'wspd850_M', 'wspd700_M', 'wspd500_M', 'Q975_M', 'Q925_M',
       'Q850_M', 'Q700_M', 'Q500_M']

obs_range_dic={'t2m_obs':[-30,42], # Official value: [-20,42]
                'rh2m_obs':[0.0,100.0],
                'w10m_obs':[0.0, 30.0],
                'psur_obs':[850,1100],
                'q2m_obs':[0,30],
                 'd10m_obs':[0.0, 360.0],
                 'u10m_obs':[-25.0, 20.0], # Official value: [-20,20]
                 'v10m_obs':[-20.0, 20.0],
                 'RAIN_obs':[0.0, 300.0],}

ruitu_range_dic={'psfc_M':[850,1100],
                't2m_M':[-30,42], # Official value: [-20,42]
                'q2m_M':[-0,30],
                 'rh2m_M':[0.0,100.0],
                 'w10m_M':[0.0, 30.0],
                 'd10m_M':[0.0, 360.0],
                 'u10m_M':[-25.0, 20.0], # Official value: [-20,20]
                 'v10m_M':[-20.0, 20.0],
                 'SWD_M':[0.0, 1200.0],
                 'GLW_M':[0.0, 550.0],
                 'HFX_M':[-200.0, 500.0],
                 'LH_M':[-50.0, 300.0],
                 'RAIN_M':[0.0, 300.0],
                 'PBLH_M':[0.0, 5200.0],
                 'TC975_M':[-30.0, 40.0],
                 'TC925_M':[-35.0, 38.0],
                 'TC850_M':[-38.0, 35.0],
                 'TC700_M':[-45.0, 30.0],
                 'TC500_M':[-70.0, 28.0],
                 'wspd975_M':[0.0, 50.0],
                 'wspd925_M':[0.0, 50.0],
                 'wspd850_M':[0.0, 50.0],
                 'wspd700_M':[0.0, 50.0],
                 'wspd500_M':[0.0, 50.0],
                 'Q975_M':[0.0, 10.0],
                 'Q925_M':[0.0, 10.0],
                 'Q850_M':[0.0, 10.0],
                 'Q700_M':[0.0, 10.0],
                 'Q500_M':[0.0, 5.0],
                }


def netCDF_filter_nan(data_file, phase_str, interim_path):
    '''
    phase_str: train, val or test
    '''
    data_dic={'input_obs': None, 'input_ruitu': None, 'ground_truth': None}

    print('processing...:', data_file)
    ori_data = nc.Dataset(data_file)     # load original data
    ori_dimensions, ori_variables= ori_data.dimensions, ori_data.variables
    date_index, fortime_index, station_index = 1, 2, 3
    var_obs = [] # var name list
    var_all =[]
    var_ruitu=[]
    for v in ori_variables:
        var_all.append(v)
        if v.find("_obs") != -1:
            var_obs.append(v)
        elif v.find('_M') != -1:
            var_ruitu.append(v)

    sta_id = ori_variables['station'][:].data
    print('sta_id:', sta_id)
    hour_index = ori_variables['foretimes'][:].data
    print('hour_index:', hour_index)
    day_index = ori_variables['date'][:].data
    print('day_index:', day_index)
    print(str(list(day_index)[-1]).split('.')[0])

    # build a map for staion and its index
    station_dic ={}
    for i,s in enumerate(sta_id):
        station_dic[s]=i
    print(station_dic)

    NUMS = ori_dimensions['date'].size # 1188 for train
    input_ruitu_nan_list=[]
    input_obs_nan_list=[]
    #output_obs_nan_list=[]
    for i in range(NUMS-1):
        input_ruitu_nan = (ori_variables['t2m_M'][i,:,:].data == -9999.).all()
        input_obs_nan = (ori_variables['t2m_obs'][i,:,:].data == -9999.).all()

        if input_ruitu_nan:
            input_ruitu_nan_list.append(i)

        if input_obs_nan:
            input_obs_nan_list.append(i)

    input_ruitu_nan_list_minus1 = [i-1 for i in input_ruitu_nan_list]
    print('input_ruitu_nan_list_minus1:', input_ruitu_nan_list_minus1)
    print('input_obs_nan_list', input_obs_nan_list)
    deleted_obs_days=input_ruitu_nan_list_minus1+input_obs_nan_list
    #bad_days
    print('deleted_obs_days:', (deleted_obs_days))
    print('deleted_obs_days_nums:', len(deleted_obs_days))
    input_obs_dic = dict.fromkeys(var_obs, None)
    input_ruitu_dic = dict.fromkeys(var_ruitu, None)
    ground_truth_dic = dict.fromkeys(var_obs, None)
    good_obs_days = [i for i in range(NUMS-1) if i not in deleted_obs_days]
    print('The number of not seriously NaN days:', len(good_obs_days))

    good_groundtruth_days = [i+1 for i in good_obs_days] # one day after observable timestamp
    print(var_obs)

    for v in var_obs:
        input_obs_dic[v] = ori_variables[v][good_obs_days,:,:].data
        ground_truth_dic[v] = ori_variables[v][good_groundtruth_days,:,:].data
    for v in var_ruitu:
        input_ruitu_dic[v] = ori_variables[v][good_groundtruth_days,:,:].data

    for v in var_obs:
        np.place(input_obs_dic[v], input_obs_dic[v]==-9999., np.nan)
        # Fill missing value with mean value
        mean_ = np.nanmean(input_obs_dic[v]) # Calculate mean except np.nan
        where_are_NaNs = np.isnan(input_obs_dic[v])
        input_obs_dic[v][where_are_NaNs]=mean_

        np.place(ground_truth_dic[v], ground_truth_dic[v]==-9999., np.nan)
        mean_ = np.nanmean(ground_truth_dic[v]) # Calculate mean except np.nan
        where_are_NaNs = np.isnan(ground_truth_dic[v])
        ground_truth_dic[v][where_are_NaNs]=mean_

    data_dic['input_obs']=input_obs_dic
    data_dic['ground_truth']=ground_truth_dic

    for v in var_ruitu:
        np.place(input_ruitu_dic[v], input_ruitu_dic[v]==-9999., np.nan)
        mean_ = np.nanmean(input_ruitu_dic[v]) # Calculate mean except np.nan
        where_are_NaNs = np.isnan(input_ruitu_dic[v])
        input_ruitu_dic[v][where_are_NaNs]=mean_

    data_dic['input_ruitu']=input_ruitu_dic
    save_pkl(data_dic, interim_path, '{}_non_NaN.dict'.format(phase_str))

    return '{}_non_NaN.dict'.format(phase_str)

def process_outlier_and_normalize(ndarray, max_min):
    '''
    Set outlier value into the normal range according to ruitu_range_dic AND obs_range_dic
    '''
    min_ = max_min[0]
    max_ = max_min[1]

    where_lower_min = ndarray < min_
    where_higher_max = ndarray > max_

    ndarray[where_lower_min]=min_
    ndarray[where_higher_max]=max_

    # move normalize to load_data function
    # ndarray = min_max_norm(ndarray, min_, max_)
    return ndarray


def process_outlier_and_stack(interim_path, file_name, phase_str, processed_path):
    data_nc = load_pkl(interim_path, file_name)
    # Outlier processing
    for v in obs_var:
        data_nc['input_obs'][v] = process_outlier_and_normalize(data_nc['input_obs'][v], obs_range_dic[v])
        data_nc['ground_truth'][v] = process_outlier_and_normalize(data_nc['ground_truth'][v], obs_range_dic[v])
    for v in ruitu_var:
        data_nc['input_ruitu'][v] = process_outlier_and_normalize(data_nc['input_ruitu'][v], ruitu_range_dic[v])

    stacked_data = [data_nc['input_obs'][v] for v in obs_var]
    stacked_input_obs = np.stack(stacked_data, axis=-1)

    stacked_data = [data_nc['input_ruitu'][v] for v in ruitu_var]
    stacked_input_ruitu = np.stack(stacked_data, axis=-1)

    stacked_data = [data_nc['ground_truth'][v] for v in obs_var]
    stacked_ground_truth = np.stack(stacked_data, axis=-1)

    print(stacked_input_obs.shape) #(sample_ind, timestep, station_id, features)
    print(stacked_input_ruitu.shape)
    print(stacked_ground_truth.shape)

    data_dic={'input_obs':stacked_input_obs,
         'input_ruitu':stacked_input_ruitu,
         'ground_truth':stacked_ground_truth}


    save_pkl(data_dic, processed_path, '{}.dict'.format(phase_str))


def main(raw_filepath, process_phase, interim_filepath, processed_path):
    """ Runs data processing scripts to turn raw data from (./data/raw) into
        cleaned data ready to be analyzed (saved in ./data/processed).
    """
    file_name = ''
    if process_phase == 'train':
        file_name = 'ai_challenger_wf2018_trainingset_20150301-20180531.nc'

    elif process_phase == 'val':
        file_name = 'ai_challenger_wf2018_validation_20180601-20180828_20180905.nc'

    elif process_phase == 'test':
        file_name = 'ai_challenger_wf2018_testb7_20180829-20181103.nc'
    interim_filename = netCDF_filter_nan(raw_filepath+file_name, process_phase, interim_filepath)
    process_outlier_and_stack(interim_filepath, interim_filename, process_phase, processed_path)


def generate_graph_seq2seq_io_data(all_obs_var, x_offsets, y_offsets, add_time_in_day=True):
    num_samples, num_nodes = all_obs_var.shape  # 样本和节点
    # data 最后扩一维
    data = np.expand_dims(all_obs_var, axis=-1)
    data_list = [data]

    if add_time_in_day:
        time_ind = [((i+3)%24) / 24 for i in range(num_samples)]# hour in day
        time_ind = np.array(time_ind)
        time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
        data_list.append(time_in_day)

    data = np.concatenate(data_list, axis=-1)
    x, y = [], []
    # t is the index of the last observation.
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive
    for t in range(min_t, max_t):
        x_t = data[t + x_offsets, ...]
        y_t = data[t + y_offsets, ...]
        x.append(x_t)
        y.append(y_t)
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    return x, y


def resize_data(interim_filepath, output_dir):
    train = load_pkl(interim_filepath, 'train.dict')
    valid = load_pkl(interim_filepath, 'val.dict')
    test = load_pkl(interim_filepath, 'test.dict')

    stacked_input_ruitu = np.concatenate((train['input_ruitu'], valid['input_ruitu'], test['input_ruitu']), axis=0)
    stacked_input_obs = np.concatenate((train['input_obs'], valid['input_obs'], test['input_obs']), axis=0)
    stacked_ground = np.concatenate((train['ground_truth'], valid['ground_truth'], test['ground_truth']), axis=0)

    print('Ruitu data shape:', stacked_input_ruitu.shape)
    print('OBS data shape:', stacked_input_obs.shape)
    print('Ground data shape:', stacked_ground.shape)

    # concat all timestamp and make 4D->3D (delete day)
    all_obs_var = stacked_input_obs[0, :, :, :] # 获得第一天3点到24点数据和第二天到15点数据
    for idx in range(1, stacked_input_obs.shape[0]):
        if idx % 2 == 0:
            all_obs_var = np.append(all_obs_var, stacked_input_obs[idx, :, :, :], axis=0)
        else:
            all_obs_var = np.append(all_obs_var, stacked_input_obs[idx, 13:-13, :, :], axis=0)
    # actually, need add final element of ground_truth, cos there is 1302 days.
    # however, ground truth only 3 target variables.
    all_obs_var = np.append(all_obs_var, stacked_ground[-1, 13:-13, :, :], axis=0)
    print(all_obs_var.shape)

    # ruitu data is different from obs, cos it aligns with ground_truth
    all_ruitu_var = stacked_input_ruitu[0, :, :, :]
    for idx in range(1, stacked_input_ruitu.shape[0]):
        if idx % 2 == 0:
            all_ruitu_var = np.append(all_ruitu_var, stacked_input_ruitu[idx, :, :, :], axis=0)
        else:
            all_ruitu_var = np.append(all_ruitu_var, stacked_input_ruitu[idx, 13:-13, :, :], axis=0)
    print(all_ruitu_var.shape)

    # delete station dimension
    all_obs_var = all_obs_var[:, 0, ]

    # horizon: 12 i.e. input:24 and out_put:12
    x_offsets = np.sort(
        np.concatenate((np.arange(-23, 1, 1),))
    )
    y_offsets = np.sort(np.arange(1, 13, 1))  # [1....12]

    x, y = generate_graph_seq2seq_io_data(
        all_obs_var,
        x_offsets=x_offsets,
        y_offsets=y_offsets,
        add_time_in_day=True
    )
    print("x shape: ", x.shape, ", y shape: ", y.shape)

    # Write the data into npz file.
    # 2 for test
    # for the rest: 7/8 is used for training, and 1/8 is used for validation.
    num_samples = all_obs_var.shape[0]
    num_test = round(num_samples * 0.2)
    num_train = round(num_samples * 0.7)
    num_val = num_samples - num_test - num_train
    print("train num: ", num_train, ", val num: ", num_val, ", test num: ", num_test)

    # train
    x_train, y_train = x[:num_train], y[:num_train]
    # val
    x_val, y_val = (
        x[num_train: num_train + num_val],
        y[num_train: num_train + num_val],
    )
    # test
    x_test, y_test = x[-num_test:], y[-num_test:]

    for cat in ["train", "val", "test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat] # local variable
        print(cat, "x: ", _x.shape, "y:", _y.shape)
        np.savez_compressed(
            os.path.join(output_dir, "%s.npz" % cat),
            x=_x,
            y=_y,
            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
        )
    print('Data saves in: ', output_dir)


if __name__ == '__main__':
    raw_filepath = './data/raw/'
    interim_filepath = './data/processed/'
    processed_path = interim_filepath

    for process_phase in ['train', 'val', 'test']:
        if not os.path.exists(interim_filepath):
            os.mkdir(interim_filepath)
        main(raw_filepath, process_phase, interim_filepath, processed_path)

    resize_data(interim_filepath, interim_filepath)