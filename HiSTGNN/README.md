# HiSTGNN
This is a Pytorch implementation of the paper: HiSTGNN: Hierarchical Graph Neural Networks for Weather Forecasting.

## Requirements
The basic dependencies are Python 3 and Torch 1.2.0. The others are specified in requirements.txt

## Model Training
#### 1. HiSTGNN
```
python train.py --hier_true True --DIL_true False --buildA_true True
```

#### 2. HiSTGNN w/o HG
```
python train.py --hier_true False --DIL_true False --buildA_true True
```

#### 3. HiSTGNN w/o DIL
```
python train.py --hier_true False --DIL_true True --buildA_true True
```

#### 4. HiSTGNN w/o ADL
```
python train.py --hier_true True --DIL_true False --buildA_true False
```
