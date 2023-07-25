import pandas as pd
import os
import sys
import numpy as np
import pickle
from pmdarima.arima import auto_arima
import itertools
from tqdm import tqdm_notebook as tqdm
from sklearn.svm import SVR
from helper import *


src_dir = os.path.join(os.getcwd(), os.pardir, 'src')
sys.path.append(src_dir)


def renorm(norm_value, min_v ,max_v):
    real_v = norm_value * (max_v-min_v) + min_v
    return real_v

def norm(values, min_v, max_v):
    norm_v = (values - min_v) / max_v
    return norm_v


processed_path = '../HiSTGNN/data/wfd_BJ'
train_data='train.dict'

obs_var=['t2m_obs','rh2m_obs','w10m_obs','psur_obs', 'q2m_obs', 'd10m_obs', 'u10m_obs', 'v10m_obs', 'RAIN_obs']
target_var=['t2m', 'rh2m', 'w10m']
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

target_range_dic={'t2m':[-30,42], # Official value: [-20,42]
                'rh2m':[0.0,100.0],
                'w10m':[0.0, 30.0]}

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

train_dict = load_pkl(processed_path, train_data)
print(train_dict.keys())
print(train_dict['input_ruitu'].shape)
print(train_dict['ground_truth'].shape)
print(train_dict['input_obs'].shape)
train_dict['input_obs'] = train_dict['input_obs'][:,:-9,:,:]
train_dict['ground_truth'] = train_dict['ground_truth'][:,4:,:,:]
print(train_dict['input_ruitu'].shape)
print(train_dict['ground_truth'].shape)
print(train_dict['input_obs'].shape)


# max-min normalize dataset
print("normalize data")

for idx, var in enumerate(obs_var):
    train_dict["input_obs"][..., idx] = norm(train_dict['input_obs'][..., idx], obs_range_dic[var][0], obs_range_dic[var][1])    

for idx, var in enumerate(ruitu_var):
    train_dict["input_ruitu"][..., idx] = norm(train_dict['input_ruitu'][..., idx], ruitu_range_dic[var][0], ruitu_range_dic[var][1])    

for idx, var in enumerate(target_var):
    train_dict["ground_truth"][..., idx] = norm(train_dict['ground_truth'][..., idx], target_range_dic[var][0], target_range_dic[var][1])  


print("Build trainable dataset.....")
X_train=[]
Y_train_t2m=[]
Y_train_rh2m=[]
Y_train_w10m=[]

sta_id= 0

# samples
for sample_id in range(1148):
    # time steps
    for time_id in range(33):
        # stations
        for sta_id in range(10):
            # import pdb; pdb.set_trace()
            fea_obs= train_dict['input_obs'][sample_id, :, sta_id, :3].reshape(-1)
            fea_M= train_dict['input_ruitu'][sample_id, time_id, sta_id,:].reshape(-1)
            # add time_id as a new feature
            fea_M= np.r_[fea_M, np.eye(33)[time_id]]
            # add staion_id as a new feature
            OneHot_ID= np.eye(10)[sta_id]
            
            label_t2m= train_dict['ground_truth'][sample_id, time_id, sta_id, 0].reshape(-1)
            label_rh2m= train_dict['ground_truth'][sample_id, time_id, sta_id, 1].reshape(-1)
            label_w10m= train_dict['ground_truth'][sample_id, time_id, sta_id, 2].reshape(-1)

            # X_train.append(np.r_[fea_obs, fea_M, OneHot_ID])
            
            # discard rui_tu dataset
            X_train.append(np.r_[fea_obs, OneHot_ID])
            
            Y_train_t2m.append(label_t2m)
            Y_train_rh2m.append(label_rh2m)
            Y_train_w10m.append(label_w10m)            

X_train=np.array(X_train)
Y_train_t2m=np.array(Y_train_t2m)
Y_train_t2m= Y_train_t2m.ravel()
Y_train_rh2m=np.array(Y_train_rh2m)
Y_train_rh2m= Y_train_rh2m.reshape(-1) 
Y_train_w10m=np.array(Y_train_w10m)
Y_train_w10m= Y_train_w10m.reshape(-1) 


print("initialize model for rh2m....")
clf_rh2m = SVR(kernel='rbf', gamma='auto', C=1.0, epsilon=0.2, verbose=2, tol=1e-2, max_iter=1000)
print('model training')
clf_rh2m.fit(X_train, Y_train_rh2m)

with open('./SVR_rh2m.pkl', 'wb') as handle:
    pickle.dump(clf_rh2m, handle, protocol=pickle.HIGHEST_PROTOCOL)

print('ok for rh2m')


print("initialize model for t2m....")
clf_t2m = SVR(gamma='auto', C=1.0, epsilon=0.2, verbose=2, tol=0.1, max_iter=1000)
print('model training')
clf_t2m.fit(X_train, Y_train_t2m)
with open('./SVR_t2m.pkl', 'wb') as handle:
    pickle.dump(clf_rh2m, handle, protocol=pickle.HIGHEST_PROTOCOL)
print('ok for t2m')


print("initialize model for w10m....")
clf_w10m = SVR(gamma='auto', C=1.0, epsilon=0.2, verbose=2, tol=0.1, max_iter=1000)
print('model training')
clf_w10m.fit(X_train, Y_train_w10m)
with open('./SVR_w10m.pkl', 'wb') as handle:
    pickle.dump(clf_w10m, handle, protocol=pickle.HIGHEST_PROTOCOL)

print('ok for w10m')


with open('./SVR_w10m.pkl', 'rb') as handle:
    clf_w10m = pickle.load(handle)

with open('./SVR_t2m.pkl', 'rb') as handle:
    clf_t2m = pickle.load(handle)

with open('./SVR_rh2m.pkl', 'rb') as handle:
    clf_rh2m = pickle.load(handle)


test_data='test.dict'
test_dict = load_pkl(processed_path, test_data)
print(test_dict['input_ruitu'].shape)
print(test_dict['ground_truth'].shape)
print(test_dict['input_ruitu'].shape)
test_dict['input_obs'] = test_dict['input_obs'][:,:-9,:,:]
test_dict['ground_truth'] = test_dict['ground_truth'][:,4:,:,:]

print("normalize test data")

for idx, var in enumerate(obs_var):
    test_dict["input_obs"][..., idx] = norm(test_dict['input_obs'][..., idx], obs_range_dic[var][0], obs_range_dic[var][1])    

for idx, var in enumerate(ruitu_var):
    test_dict["input_ruitu"][..., idx] = norm(test_dict['input_ruitu'][..., idx], ruitu_range_dic[var][0], ruitu_range_dic[var][1])    

ground_t2m = norm(test_dict['ground_truth'][..., 0], target_range_dic['t2m'][0], target_range_dic['t2m'][1])
ground_rh2m = norm(test_dict['ground_truth'][..., 1], target_range_dic['rh2m'][0], target_range_dic['rh2m'][1])
ground_w10m = norm(test_dict['ground_truth'][..., 2], target_range_dic['w10m'][0], target_range_dic['w10m'][1])


X_test=[]
Y_test_t2m = []
Y_test_rh2m = []
Y_test_w10m = []

for i in range(test_dict['input_obs'].shape[0]):
    for sta_id in range(10):
        for time_id in range(33):
            # import pdb;pdb.set_trace()
            fea_obs= test_dict['input_obs'][i, :, sta_id,:3].reshape(-1)
            fea_M= test_dict['input_ruitu'][i, time_id, sta_id,:].reshape(-1)
            # add time_id as a new feature
            fea_M= np.r_[fea_M, np.eye(33)[time_id]]
            # add staion_id as a new feature
            OneHot_ID= np.eye(10)[sta_id]
            X_test.append(np.r_[fea_obs, OneHot_ID])

            label_t2m = ground_t2m[i, time_id, sta_id].reshape(-1)
            label_rh2m = ground_rh2m[i, time_id, sta_id].reshape(-1)
            label_w10m = ground_w10m[i, time_id, sta_id].reshape(-1)

            Y_test_t2m.append(label_t2m)
            Y_test_rh2m.append(label_rh2m)
            Y_test_w10m.append(label_w10m)


X_test=np.array(X_test)
Y_test_t2m=np.array(Y_test_t2m)
Y_test_rh2m=np.array(Y_test_rh2m)
Y_test_w10m=np.array(Y_test_w10m)


Y_rh2m_pred= clf_rh2m.predict(X_test)
Y_t2m_pred= clf_t2m.predict(X_test)
Y_w10m_pred= clf_w10m.predict(X_test)

print('ok')

Y_rh2m_submit = renorm(Y_rh2m_pred, target_range_dic['rh2m'][0], target_range_dic['rh2m'][1])
Y_t2m_submit = renorm(Y_t2m_pred, target_range_dic['t2m'][0], target_range_dic['t2m'][1])
Y_w10m_submit = renorm(Y_w10m_pred, target_range_dic['w10m'][0], target_range_dic['w10m'][1])

Y_test_rh2m = renorm(Y_test_rh2m, target_range_dic['rh2m'][0], target_range_dic['rh2m'][1])
Y_test_t2m = renorm(Y_test_t2m, target_range_dic['t2m'][0], target_range_dic['t2m'][1])
Y_test_w10m = renorm(Y_test_w10m, target_range_dic['w10m'][0], target_range_dic['w10m'][1])


with open("./svr_rh2m_BJ.npy", "wb") as f:
    np.save(f, Y_rh2m_submit)


with open("./svr_t2m_BJ.npy", "wb") as f:
    np.save(f, Y_t2m_submit)


with open("./svr_w10m_BJ.npy", "wb") as f:
    np.save(f, Y_w10m_submit)


# ------------------for USA dataset-------------------------------
USA_range = [(-35.0, 40.0), (0.0, 40.0), (800.0, 1100.0), (0.0, 100.0)]
USA = ['Boston', 'New York', 'Philadelphia', 'Detroit', 'Pittsburgh', 'Chicago', 'Indianapolis', 'Charlotte', 'Saint Louis', 'Nashville', 'Atlanta', 'Jacksonville', 'Miami']
USA_DATA_Dir = "../HiSTGNN/data/wfd_USA/"

ori_data = load_pkl(USA_DATA_Dir, 'USA_data.dict')
# (samples, time, variable, station)

for category in ['train', 'val', 'test']:
    for idx, var in enumerate(USA_range):
        ori_data['x_'+category][:, :, idx, :] = norm(ori_data['x_'+category][:, :, idx, :], USA_range[idx][0], USA_range[idx][1])
        ori_data['y_'+category][:, :, idx, :] = norm(ori_data['y_'+category][:, :, idx, :], USA_range[idx][0], USA_range[idx][1])


print("Build trainable dataset.....")
X_train=[]
Y_train_t2m=[]
Y_train_w10m=[]
Y_train_psur=[]
Y_train_rh2m=[]

sta_id= 0

# samples
for sample_id in range(ori_data['x_train'].shape[0]):
    # time steps
    for time_id in range(ori_data['y_train'].shape[1]):
        # stations
        for sta_id in range(ori_data['x_train'].shape[-1]):
            # import pdb; pdb.set_trace()
            fea_obs= ori_data['x_train'][sample_id, :, :, sta_id].reshape(-1)
            # add staion_id as a new feature
            OneHot_ID= np.eye(13)[sta_id]
            
            label_t2m= ori_data['y_train'][sample_id, time_id, 0, sta_id].reshape(-1)
            label_w10m= ori_data['y_train'][sample_id, time_id, 1, sta_id].reshape(-1)
            label_psur= ori_data['y_train'][sample_id, time_id, 2, sta_id].reshape(-1)
            label_rh2m= ori_data['y_train'][sample_id, time_id, 3, sta_id,].reshape(-1)
            
            # discard rui_tu dataset
            X_train.append(np.r_[fea_obs, OneHot_ID])
            
            Y_train_t2m.append(label_t2m)
            Y_train_psur.append(label_psur)
            Y_train_rh2m.append(label_rh2m)
            Y_train_w10m.append(label_w10m)  
            
for sample_id in range(ori_data['x_val'].shape[0]):
    # time steps
    for time_id in range(ori_data['y_val'].shape[1]):
        # stations
        for sta_id in range(ori_data['x_val'].shape[-1]):
            # import pdb; pdb.set_trace()
            fea_obs= ori_data['x_val'][sample_id, :, :, sta_id].reshape(-1)
            # add staion_id as a new feature
            OneHot_ID= np.eye(13)[sta_id]
            
            label_t2m= ori_data['y_val'][sample_id, time_id, 0, sta_id].reshape(-1)
            label_w10m= ori_data['y_val'][sample_id, time_id, 1, sta_id].reshape(-1)
            label_psur= ori_data['y_val'][sample_id, time_id, 2, sta_id].reshape(-1)
            label_rh2m= ori_data['y_val'][sample_id, time_id, 3, sta_id,].reshape(-1)
            
            # discard rui_tu dataset
            X_train.append(np.r_[fea_obs, OneHot_ID])
            
            Y_train_t2m.append(label_t2m)
            Y_train_psur.append(label_psur)
            Y_train_rh2m.append(label_rh2m)
            Y_train_w10m.append(label_w10m)          


X_train=np.array(X_train)
Y_train_t2m=np.array(Y_train_t2m)
Y_train_t2m= Y_train_t2m.ravel()
Y_train_rh2m=np.array(Y_train_rh2m)
Y_train_rh2m= Y_train_rh2m.reshape(-1) 
Y_train_w10m=np.array(Y_train_w10m)
Y_train_w10m= Y_train_w10m.reshape(-1)
Y_train_psur=np.array(Y_train_psur)
Y_train_psur= Y_train_psur.reshape(-1) 


print("initialize model for rh2m....")
clf_rh2m = SVR(kernel='rbf', gamma='auto', C=1.0, epsilon=0.2, verbose=2, tol=1e-2, max_iter=1000)
print('model training')
clf_rh2m.fit(X_train, Y_train_rh2m)

with open('./USA_SVR_rh2m.pkl', 'wb') as handle:
    pickle.dump(clf_rh2m, handle, protocol=pickle.HIGHEST_PROTOCOL)

print('ok for rh2m')


print("initialize model for t2m....")
clf_t2m = SVR(gamma='auto', C=1.0, epsilon=0.2, verbose=2, tol=0.1, max_iter=1000)
print('model training')
clf_t2m.fit(X_train, Y_train_t2m)
with open('./USA_SVR_t2m.pkl', 'wb') as handle:
    pickle.dump(clf_rh2m, handle, protocol=pickle.HIGHEST_PROTOCOL)
print('ok for t2m')


print("initialize model for w10m....")
clf_w10m = SVR(gamma='auto', C=1.0, epsilon=0.2, verbose=2, tol=0.1, max_iter=1000)
print('model training')
clf_w10m.fit(X_train, Y_train_w10m)
with open('./USA_SVR_w10m.pkl', 'wb') as handle:
    pickle.dump(clf_w10m, handle, protocol=pickle.HIGHEST_PROTOCOL)

print('ok for w10m')

print("initialize model for pressure....")
clf_psur = SVR(gamma='auto', C=1.0, epsilon=0.2, verbose=2, tol=0.1, max_iter=1000)
print('model training')
clf_psur.fit(X_train, Y_train_psur)
with open('./USA_SVR_psur.pkl', 'wb') as handle:
    pickle.dump(clf_psur, handle, protocol=pickle.HIGHEST_PROTOCOL)

print('ok for psur')


with open('./USA_SVR_w10m.pkl', 'rb') as handle:
    clf_w10m = pickle.load(handle)

with open('./USA_SVR_t2m.pkl', 'rb') as handle:
    clf_t2m = pickle.load(handle)

with open('./USA_SVR_rh2m.pkl', 'rb') as handle:
    clf_rh2m = pickle.load(handle)

with open('./USA_SVR_psur.pkl', 'rb') as handle:
    clf_psur = pickle.load(handle)
    
    
X_test=[]
Y_test_t2m=[]
Y_test_w10m=[]
Y_test_psur=[]
Y_test_rh2m=[]

for sample_id in range(ori_data['x_test'].shape[0]):
    # time steps
    for time_id in range(ori_data['y_test'].shape[1]):
        # stations
        for sta_id in range(ori_data['x_test'].shape[-1]):
            # import pdb; pdb.set_trace()
            fea_obs= ori_data['x_test'][sample_id, :, :, sta_id].reshape(-1)
            # add staion_id as a new feature
            OneHot_ID= np.eye(13)[sta_id]
            
            label_t2m= ori_data['y_test'][sample_id, time_id, 0, sta_id].reshape(-1)
            label_w10m= ori_data['y_test'][sample_id, time_id, 1, sta_id].reshape(-1)
            label_psur= ori_data['y_test'][sample_id, time_id, 2, sta_id].reshape(-1)
            label_rh2m= ori_data['y_test'][sample_id, time_id, 3, sta_id,].reshape(-1)
            
            # discard rui_tu dataset
            X_test.append(np.r_[fea_obs, OneHot_ID])
            
            Y_test_t2m.append(label_t2m)
            Y_test_psur.append(label_psur)
            Y_test_rh2m.append(label_rh2m)
            Y_test_w10m.append(label_w10m)

X_test=np.array(X_test)
Y_test_t2m=np.array(Y_test_t2m)
Y_test_t2m= Y_test_t2m.ravel()
Y_test_rh2m=np.array(Y_test_rh2m)
Y_test_rh2m= Y_test_rh2m.reshape(-1) 
Y_test_w10m=np.array(Y_test_w10m)
Y_test_w10m= Y_test_w10m.reshape(-1)
Y_test_psur=np.array(Y_test_psur)
Y_test_psur= Y_test_psur.reshape(-1) 


Y_rh2m_pred= clf_rh2m.predict(X_test)
Y_t2m_pred= clf_t2m.predict(X_test)
Y_w10m_pred= clf_w10m.predict(X_test)
Y_psur_pred= clf_psur.predict(X_test)

print('ok')


Y_t2m_submit = renorm(Y_t2m_pred, USA_range[0][0], USA_range[0][1])
Y_w10m_submit = renorm(Y_w10m_pred, USA_range[1][0], USA_range[1][1])
Y_psur_submit = renorm(Y_psur_pred, USA_range[2][0], USA_range[2][1])
Y_rh2m_submit = renorm(Y_rh2m_pred, USA_range[3][0], USA_range[3][1])


Y_test_t2m = renorm(Y_test_t2m, USA_range[0][0], USA_range[0][1])
Y_test_w10m = renorm(Y_test_w10m, USA_range[1][0], USA_range[1][1])
Y_test_psur = renorm(Y_test_psur, USA_range[2][0], USA_range[2][1])
Y_test_rh2m = renorm(Y_test_rh2m, USA_range[3][0], USA_range[3][1])


with open("./svr_rh2m_USA.npy", "wb") as f:
    np.save(f, Y_rh2m_submit)


with open("./svr_t2m_USA.npy", "wb") as f:
    np.save(f, Y_t2m_submit)


with open("./svr_w10m_USA.npy", "wb") as f:
    np.save(f, Y_w10m_submit)
    

with open("./svr_pusr_USA.npy", "wb") as f:
    np.save(f, Y_psur_submit)
    


# ------------------for Israel dataset-------------------------------
Israel_range = [(-20.0, 50.0), (0.0, 60.0), (900.0, 1100.0), (0.0, 100.0)]

DATA_Dir = "../HiSTGNN/data/wfd_Israel/"

ori_data = load_pkl(DATA_Dir, 'Israel_data.dict')
# (samples, time, variable, station)
for category in ['train', 'val', 'test']:
    for idx, var in enumerate(Israel_range):
        ori_data['x_'+category][:, :, idx, :] = norm(ori_data['x_'+category][:, :, idx, :], Israel_range[idx][0], Israel_range[idx][1])
        ori_data['y_'+category][:, :, idx, :] = norm(ori_data['y_'+category][:, :, idx, :], Israel_range[idx][0], Israel_range[idx][1])


print("Build trainable dataset.....")
X_train=[]
Y_train_t2m=[]
Y_train_w10m=[]
Y_train_psur=[]
Y_train_rh2m=[]

sta_id= 0

'''
X: [[feature], [], []]  samples x stations x future_step
Y: [1, ..., 24, 1, ...., 24]   if the lenght of output is 24, 依次展开为所有样本、站点、所有解码步长
'''
# samples
for sample_id in range(ori_data['x_train'].shape[0]):
    # time steps
    for time_id in range(ori_data['y_train'].shape[1]):
        # stations flatten
        for sta_id in range(ori_data['x_train'].shape[-1]):
            # import pdb; pdb.set_trace()
            fea_obs= ori_data['x_train'][sample_id, :, :, sta_id].reshape(-1)
            # add staion_id as a new feature
            OneHot_ID= np.eye(6)[sta_id]
            
            label_t2m= ori_data['y_train'][sample_id, time_id, 0, sta_id].reshape(-1)
            label_w10m= ori_data['y_train'][sample_id, time_id, 1, sta_id].reshape(-1)
            label_psur= ori_data['y_train'][sample_id, time_id, 2, sta_id].reshape(-1)
            label_rh2m= ori_data['y_train'][sample_id, time_id, 3, sta_id,].reshape(-1)
            
            # discard rui_tu dataset
            X_train.append(np.r_[fea_obs, OneHot_ID])
            
            Y_train_t2m.append(label_t2m)
            Y_train_psur.append(label_psur)
            Y_train_rh2m.append(label_rh2m)
            Y_train_w10m.append(label_w10m)  
            
for sample_id in range(ori_data['x_val'].shape[0]):
    # time steps
    for time_id in range(ori_data['y_val'].shape[1]):
        # stations
        for sta_id in range(ori_data['x_val'].shape[-1]):
            # import pdb; pdb.set_trace()
            fea_obs= ori_data['x_val'][sample_id, :, :, sta_id].reshape(-1)
            # add staion_id as a new feature
            OneHot_ID= np.eye(6)[sta_id]
            
            label_t2m= ori_data['y_val'][sample_id, time_id, 0, sta_id].reshape(-1)
            label_w10m= ori_data['y_val'][sample_id, time_id, 1, sta_id].reshape(-1)
            label_psur= ori_data['y_val'][sample_id, time_id, 2, sta_id].reshape(-1)
            label_rh2m= ori_data['y_val'][sample_id, time_id, 3, sta_id,].reshape(-1)
            
            # discard rui_tu dataset
            X_train.append(np.r_[fea_obs, OneHot_ID])
            
            Y_train_t2m.append(label_t2m)
            Y_train_psur.append(label_psur)
            Y_train_rh2m.append(label_rh2m)
            Y_train_w10m.append(label_w10m)          

# (samples, features)
X_train=np.array(X_train)
Y_train_t2m=np.array(Y_train_t2m)
Y_train_t2m= Y_train_t2m.reshape(-1)
Y_train_rh2m=np.array(Y_train_rh2m)
Y_train_rh2m= Y_train_rh2m.reshape(-1) 
Y_train_w10m=np.array(Y_train_w10m)
Y_train_w10m= Y_train_w10m.reshape(-1)
Y_train_psur=np.array(Y_train_psur)
Y_train_psur= Y_train_psur.reshape(-1)


print("initialize model for rh2m....")
clf_rh2m = SVR(kernel='rbf', gamma='auto', C=1.0, epsilon=0.2, verbose=2, tol=1e-2, max_iter=1000)
print('model training')
clf_rh2m.fit(X_train, Y_train_rh2m)

with open('./Israel_SVR_rh2m.pkl', 'wb') as handle:
    pickle.dump(clf_rh2m, handle, protocol=pickle.HIGHEST_PROTOCOL)

print('ok for rh2m')


print("initialize model for t2m....")
clf_t2m = SVR(gamma='auto', C=1.0, epsilon=0.2, verbose=2, tol=0.1, max_iter=1000)
print('model training')
clf_t2m.fit(X_train, Y_train_t2m)
with open('./Israel_SVR_t2m.pkl', 'wb') as handle:
    pickle.dump(clf_rh2m, handle, protocol=pickle.HIGHEST_PROTOCOL)
print('ok for t2m')


print("initialize model for w10m....")
clf_w10m = SVR(gamma='auto', C=1.0, epsilon=0.2, verbose=2, tol=0.1, max_iter=1000)
print('model training')
clf_w10m.fit(X_train, Y_train_w10m)
with open('./Israel_SVR_w10m.pkl', 'wb') as handle:
    pickle.dump(clf_w10m, handle, protocol=pickle.HIGHEST_PROTOCOL)

print('ok for w10m')

print("initialize model for pressure....")
clf_psur = SVR(gamma='auto', C=1.0, epsilon=0.2, verbose=2, tol=0.1, max_iter=1000)
print('model training')
clf_psur.fit(X_train, Y_train_psur)
with open('./Israel_SVR_psur.pkl', 'wb') as handle:
    pickle.dump(clf_psur, handle, protocol=pickle.HIGHEST_PROTOCOL)

print('ok for psur')


with open('./Israel_SVR_w10m.pkl', 'rb') as handle:
    clf_w10m = pickle.load(handle)

with open('./Israel_SVR_t2m.pkl', 'rb') as handle:
    clf_t2m = pickle.load(handle)

with open('./Israel_SVR_rh2m.pkl', 'rb') as handle:
    clf_rh2m = pickle.load(handle)

with open('./Israel_SVR_psur.pkl', 'rb') as handle:
    clf_psur = pickle.load(handle)
    
    
X_test=[]
Y_test_t2m=[]
Y_test_w10m=[]
Y_test_psur=[]
Y_test_rh2m=[]

for sample_id in range(ori_data['x_test'].shape[0]):
    # time steps
    for time_id in range(ori_data['y_test'].shape[1]):
        # stations
        for sta_id in range(ori_data['x_test'].shape[-1]):
            # import pdb; pdb.set_trace()
            fea_obs= ori_data['x_test'][sample_id, :, :, sta_id].reshape(-1)
            # add staion_id as a new feature
            OneHot_ID= np.eye(6)[sta_id]
            
            label_t2m= ori_data['y_test'][sample_id, time_id, 0, sta_id].reshape(-1)
            label_w10m= ori_data['y_test'][sample_id, time_id, 1, sta_id].reshape(-1)
            label_psur= ori_data['y_test'][sample_id, time_id, 2, sta_id].reshape(-1)
            label_rh2m= ori_data['y_test'][sample_id, time_id, 3, sta_id,].reshape(-1)
            
            # discard rui_tu dataset
            X_test.append(np.r_[fea_obs, OneHot_ID])
            
            Y_test_t2m.append(label_t2m)
            Y_test_psur.append(label_psur)
            Y_test_rh2m.append(label_rh2m)
            Y_test_w10m.append(label_w10m)

X_test=np.array(X_test)
Y_test_t2m=np.array(Y_test_t2m)
Y_test_t2m= Y_test_t2m.reshape(-1)
Y_test_rh2m=np.array(Y_test_rh2m)
Y_test_rh2m= Y_test_rh2m.reshape(-1) 
Y_test_w10m=np.array(Y_test_w10m)
Y_test_w10m= Y_test_w10m.reshape(-1)
Y_test_psur=np.array(Y_test_psur)
Y_test_psur= Y_test_psur.reshape(-1) 


Y_rh2m_pred= clf_rh2m.predict(X_test)
Y_t2m_pred= clf_t2m.predict(X_test)
Y_w10m_pred= clf_w10m.predict(X_test)
Y_psur_pred= clf_psur.predict(X_test)
print('ok')

Y_t2m_submit = renorm(Y_t2m_pred, Israel_range[0][0], Israel_range[0][1])
Y_w10m_submit = renorm(Y_w10m_pred, Israel_range[1][0], Israel_range[1][1])
Y_psur_submit = renorm(Y_psur_pred, Israel_range[2][0], Israel_range[2][1])
Y_rh2m_submit = renorm(Y_rh2m_pred, Israel_range[3][0], Israel_range[3][1])

Y_test_t2m = renorm(Y_test_t2m, Israel_range[0][0], Israel_range[0][1])
Y_test_w10m = renorm(Y_test_w10m, Israel_range[1][0], Israel_range[1][1])
Y_test_psur = renorm(Y_test_psur, Israel_range[2][0], Israel_range[2][1])
Y_test_rh2m = renorm(Y_test_rh2m, Israel_range[3][0], Israel_range[3][1])



with open("./svr_rh2m_Israel.npy", "wb") as f:
    np.save(f, Y_rh2m_submit)


with open("./svr_t2m_Israel.npy", "wb") as f:
    np.save(f, Y_t2m_submit)


with open("./svr_w10m_Israel.npy", "wb") as f:
    np.save(f, Y_w10m_submit)
    

with open("./svr_pusr_Israel.npy", "wb") as f:
    np.save(f, Y_psur_submit)