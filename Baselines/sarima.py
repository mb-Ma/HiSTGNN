import pandas as pd
import os
import sys
import numpy as np
import pickle
from pmdarima.arima import auto_arima
import itertools
from tqdm import tqdm as tqdm
import sys

from helper import *

# load BJ dataset

processed_path = '../HiSTGNN/data/wfd_BJ'
train_data='train.dict'
val_data = 'val.dict'
test_data = 'test.dict'

obs_var=['t2m_obs','rh2m_obs','w10m_obs','psur_obs', 'q2m_obs', 'd10m_obs', 'u10m_obs', 'v10m_obs', 'RAIN_obs']
target_var=['t2m_', 'rh2m_', 'w10m_']
ruitu_var=['psfc_M', 't2m_M', 'q2m_M', 'rh2m_M', 'w10m_M', 'd10m_M', 'u10m_M',
       'v10m_M', 'SWD_M', 'GLW_M', 'HFX_M', 'LH_M', 'RAIN_M', 'PBLH_M',
       'TC975_M', 'TC925_M', 'TC850_M', 'TC700_M', 'TC500_M', 'wspd975_M',
       'wspd925_M', 'wspd850_M', 'wspd700_M', 'wspd500_M', 'Q975_M', 'Q925_M',
       'Q850_M', 'Q700_M', 'Q500_M']

print('training data....')
train_dict = load_pkl(processed_path, train_data)
print(train_dict.keys())
print(train_dict['input_obs'].shape)
print(train_dict['input_ruitu'].shape)
print(train_dict['ground_truth'].shape)

print('valid data....')
val_dict = load_pkl(processed_path, val_data)
print(val_dict.keys())
print(val_dict['input_obs'].shape)
print(val_dict['input_ruitu'].shape)
print(val_dict['ground_truth'].shape)

print('test data....')
test_dict = load_pkl(processed_path, test_data)
print(test_dict.keys())
print(test_dict['input_obs'].shape)
print(test_dict['input_ruitu'].shape)
print(test_dict['ground_truth'].shape)



# build train_set and test_set
"""
Taking the training set as an example, we delete 40 days with block missing values from a total of 1188 days,
leving the trianing data from 1148 days. 

"""
# 2015/3/1-2018/5/31, 2018/6/1-2018/8/28, 2018/8/29-2018/11/3
# choose 2017/8/28 - 2018/8/28 


# build train samples
start_time=pd.datetime(2018, 8, 10)
end_time= pd.datetime(2018, 11, 3) # 37 hours after 20181028
# first truncate the day-level period
time_interval= pd.date_range(start_time, end_time, freq='D')
train = np.concatenate((train_dict['input_obs'], val_dict['input_obs'], test_dict['input_obs']), axis=0)
train = train[-len(time_interval):, ...]
train = np.float16(train)


# build consecutive time
# first day includes 3 am - 2 am (next day)
t2m_samples = [train[0, :24, :, 0]]
rh2m_samples = [train[0, :24, :, 1]]
w10m_samples = [train[0, :24, :, 2]]

# concatenate the remaining data, but not the last day
for i in range(1, train.shape[0]-1):
    # 3 am - 2 am, ending day is 11.3 2 am
    t2m_samples.append(train[i, :24, :, 0])
    rh2m_samples.append(train[i, :24, :, 1])
    w10m_samples.append(train[i, :24, :, 2])

# appending 11.3 3 am - 11.4 6 am
t2m_samples.append(train[-1, :-9, :, 0])
rh2m_samples.append(train[-1, :-9, :, 1])
w10m_samples.append(train[-1, :-9, :, 2])

# obtain final train data
start_time = pd.datetime(2018, 8, 10, 3)
end_time = pd.datetime(2018, 11, 4, 6)

date = pd.date_range(start=start_time, end=end_time, freq='H')
# time end_day 2018 08 28

t2m_samples = pd.DataFrame(np.concatenate(t2m_samples, axis=0), index=date)
rh2m_samples = pd.DataFrame(np.concatenate(rh2m_samples, axis=0), index=date)
w10m_samples = pd.DataFrame(np.concatenate(w10m_samples, axis=0), index=date)
# import pdb; pdb.set_trace()

start_time = pd.datetime(2018, 8, 29, 3)
end_time = pd.datetime(2018, 8, 30, 6)

pred = []
for idx in range(test_dict['input_obs'].shape[0]):
    # input 3 am - 6 am, output, 7 am - 3 pm (8/31)
    # add each test_day, target period 7 am - 3pm
    
    # foreach stations
    pred_stat = []
    for i in tqdm(range(10)):
        print('station: {} training for t2m ...'.format(i))
        # import pdb; pdb.set_trace()
        modl = auto_arima(t2m_samples[i][start_time:end_time], start_p=1, start_q=1, start_P=1, start_Q=1,
                  max_p=5, max_q=5, max_P=5, max_Q=5, seasonal=True,m=12,
                  stepwise=False, suppress_warnings=True, test='adf', d=None, D=1,trace=True,
                  error_action='ignore')
        print("best model: ", modl)
        hat_y, _ = modl.predict(n_periods=33, return_conf_int=True)
        pred_stat.append(hat_y)
    
    t2m_one_day = np.array(pred_stat) # 10 x 33
    
    pred_stat = []
    for i in tqdm(range(10)):
        print('station: {} training for rh2m ...'.format(i))
        modl = auto_arima(rh2m_samples[i][start_time:end_time], start_p=1, start_q=1, start_P=1, start_Q=1,
                  max_p=5, max_q=5, max_P=5, max_Q=5, seasonal=True,m=12,
                  stepwise=False, suppress_warnings=True, test='adf', d=None, D=1,
                  error_action='ignore')
        print("best model: ", modl)
        hat_y, _ = modl.predict(n_periods=33, return_conf_int=True)
        pred_stat.append(hat_y)
    
    rh2m_one_day = np.array(pred_stat) # 10 x 33
    
    pred_stat = []
    for i in tqdm(range(10)):
        print('station: {} training for w10m ...'.format(i))
        modl = auto_arima(w10m_samples[i][start_time:end_time], start_p=1, start_q=1, start_P=1, start_Q=1,
                  max_p=5, max_q=5, max_P=5, max_Q=5, seasonal=True,m=12,
                  stepwise=False, suppress_warnings=True, test='adf', d=None, D=1,
                  error_action='ignore')
        modl = auto_arima(w10m_samples[i][start_time:end_time], seasonal=True, m=12)
        print("best model: ", modl)
        hat_y, _ = modl.predict(n_periods=33, return_conf_int=True)
        pred_stat.append(hat_y)
    
    w10m_one_day = np.array(pred_stat) # 10 x 33
    
    pred.append(np.stack([t2m_one_day, rh2m_one_day, w10m_one_day], axis=2))
    end_time = end_time + pd.Timedelta(1, 'd')
    start_time = start_time + pd.Timedelta(1, 'd')
    
pred = np.array(pred)

with open('./arima_bj.npy', 'wb') as f:
    np.save(f, pred)