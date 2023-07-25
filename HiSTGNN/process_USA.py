import os
import numpy as np
from helper import save_pkl, load_pkl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

horizon = 24
windows = 48
interval = 24

# temp, Wind, presure, humidity
USA_range = [(-35.0, 40.0), (0.0, 40.0), (800.0, 1100.0), (0.0, 100.0)]

USA = ['Boston', 'New York', 'Philadelphia', 'Detroit', 'Pittsburgh', 'Chicago', 'Indianapolis', 'Charlotte', 'Saint Louis', 'Nashville', 'Atlanta', 'Jacksonville', 'Miami']


def main(path):
    pressure = pd.read_csv(os.path.join(path, 'pressure.csv'))
    pressure['datetime'] = pd.to_datetime(pressure['datetime'], format='%Y-%m-%d %H:%M:%S')
    pressure.set_index(['datetime'], inplace=True)

    wind_speed = pd.read_csv(os.path.join(path, 'wind_speed.csv'))
    wind_speed['datetime'] = pd.to_datetime(wind_speed['datetime'], format='%Y-%m-%d %H:%M:%S')
    wind_speed.set_index(['datetime'], inplace=True)

    temperature = pd.read_csv(os.path.join(path, 'temperature.csv'))
    temperature['datetime'] = pd.to_datetime(temperature['datetime'], format='%Y-%m-%d %H:%M:%S')
    temperature.set_index(['datetime'], inplace=True)

    humidity = pd.read_csv(os.path.join(path, 'humidity.csv'))
    humidity['datetime'] = pd.to_datetime(humidity['datetime'], format='%Y-%m-%d %H:%M:%S')
    humidity.set_index(['datetime'], inplace=True)

    # 将开式温度转为摄氏温度
    temperature = temperature - 273.15

    # 选择城市
    USA_humidity = humidity[USA]
    USA_temperature = temperature[USA]
    USA_windspeed = wind_speed[USA]
    USA_pressure = pressure[USA]

    # 线性插值
    USA_temperature = USA_temperature.interpolate()
    USA_windspeed = USA_windspeed.interpolate()
    USA_pressure = USA_pressure.interpolate()
    USA_humidity = USA_humidity.interpolate()

    # 截取时间
    start = '2012-10-02'
    end = '2017-10-28'
    USA_temperature = USA_temperature[start:end]
    USA_windspeed = USA_windspeed[start:end]
    USA_pressure = USA_pressure[start:end]
    USA_humidity = USA_humidity[start:end]

    # 控制数据精度
    USA_temperature = USA_temperature.round(1)
    USA_windspeed = USA_windspeed.round(1)
    USA_pressure = USA_pressure.round(1)
    USA_humidity = USA_humidity.round(1)

    # 合并数据，堆叠数据
    USA_data = []
    # 在第三个维度
    USA_data.append(USA_temperature.values)
    USA_data.append(USA_windspeed.values)
    USA_data.append(USA_pressure.values)
    USA_data.append(USA_humidity.values)
    # time, city -> time, variable, city
    USA_data = np.stack(USA_data, axis=1)

    # 组织数据
    X, Y = [], []
    for idx in range(0, USA_data.shape[0]-horizon-windows, interval):
        x = USA_data[idx:(idx + windows), ...]
        y = USA_data[(idx + windows):(idx + windows + horizon), ...]
        X.append(x)
        Y.append(y)
    X = np.array(X)
    Y = np.array(Y)

    train_rate = 0.8
    valid_rate = 0.1
    test_rate = 0.1

    train_num = int(train_rate * X.shape[0])
    test_num = int(test_rate * X.shape[0])
    valid_num = X.shape[0] - train_num - test_num

    data = {}
    data['x_train'] = X[:train_num, ...]
    data['y_train'] = Y[:train_num, ...]

    data['x_val'] = X[train_num: train_num + valid_num, ...]
    data['y_val'] = Y[train_num: train_num + valid_num, ...]

    data['x_test'] = X[train_num + valid_num:, ...]
    data['y_test'] = Y[train_num + valid_num:, ...]

    # save data
    save_pkl(data, path, 'USA_data.dict')


if __name__ == '__main__':
    path = './data/kaggle'
    main(path)