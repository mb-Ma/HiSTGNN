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

Israel_range = [(-20.0, 50.0), (0.0, 60.0), (900.0, 1100.0), (0.0, 100.0)]


def main(path):
    city = pd.read_csv(os.path.join(path, 'city_attributes.csv'))
    Canada_city = city[city['Country']=='Canada']['City'].unique()
    Israel_city = city[city['Country']=='Israel']['City'].unique()
    US_city = city[city['Country']=='United States']['City'].unique()

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

    wind_direction = pd.read_csv(os.path.join(path, 'wind_direction.csv'))
    wind_direction['datetime'] = pd.to_datetime(wind_direction['datetime'], format='%Y-%m-%d %H:%M:%S')
    wind_direction.set_index(['datetime'], inplace=True)

    # 将开式温度转为摄氏温度
    temperature = temperature - 273.15

    # 选择城市
    Israel_humidity = humidity[Israel_city]
    Israel_temperature = temperature[Israel_city]
    Israel_windspeed = wind_speed[Israel_city]
    Israel_pressure = pressure[Israel_city]

    # 线性插值
    Israel_temperature = Israel_temperature.interpolate()
    Israel_windspeed = Israel_windspeed.interpolate()
    Israel_pressure = Israel_pressure.interpolate()
    Israel_humidity = Israel_humidity.interpolate()

    # 截取时间
    start = '2012-10-02'
    end = '2017-10-28'
    Israel_temperature = Israel_temperature[start:end]
    Israel_windspeed = Israel_windspeed[start:end]
    Israel_pressure = Israel_pressure[start:end]
    Israel_humidity = Israel_humidity[start:end]

    # 控制数据精度
    Israel_temperature = Israel_temperature.round(1)
    Israel_windspeed = Israel_windspeed.round(1)
    Israel_pressure = Israel_pressure.round(1)
    Israel_humidity = Israel_humidity.round(1)

    # 合并数据，堆叠数据
    Israel_data = []
    # 在第三个维度
    Israel_data.append(Israel_temperature.values)
    Israel_data.append(Israel_windspeed.values)
    Israel_data.append(Israel_pressure.values)
    Israel_data.append(Israel_humidity.values)
    # time, city -> time, variable, city
    Israel_data = np.stack(Israel_data, axis=1)

    # 组织数据
    X, Y = [], []
    for idx in range(0, Israel_data.shape[0]-horizon-windows, interval):
        x = Israel_data[idx:(idx + windows), ...]
        y = Israel_data[(idx + windows):(idx + windows + horizon), ...]
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
    save_pkl(data, path, 'Israel_data.dict')


if __name__ == '__main__':
    path = './data/kaggle'
    main(path)