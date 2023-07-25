# python BJ_train.py --layers 3 --var_node_dim 20 --stat_node_dim 20 --batch_size 8 --tanhalpha 3         4.4396
# python BJ_train.py --layers 3 --var_node_dim 20 --stat_node_dim 20 --batch_size 16 --tanhalpha 3        4.2879
# python BJ_train.py --layers 3 --var_node_dim 20 --stat_node_dim 20 --batch_size 32 --tanhalpha 3        4.4139
# python BJ_train.py --layers 3 --var_node_dim 20 --stat_node_dim 20 --batch_size 64 --tanhalpha 3        4.2869
# python BJ_train.py --layers 3 --var_node_dim 20 --stat_node_dim 25 --batch_size 64 --tanhalpha 3        4.3466
# python BJ_train.py --layers 3 --var_node_dim 20 --stat_node_dim 20 --batch_size 128 --tanhalpha 3         4.4311

# python BJ_train.py --layers 3 --var_node_dim 5 --stat_node_dim 10 --batch_size 16 --tanhalpha 3         4.4572
# python BJ_train.py --layers 3 --var_node_dim 10 --stat_node_dim 20 --batch_size 16 --tanhalpha 3          4.3094

# python BJ_train.py --layers 3 --var_node_dim 5 --stat_node_dim 5 --batch_size 16 --tanhalpha 2                  4.3627
# python BJ_train.py --layers 3 --var_node_dim 5 --stat_node_dim 5 --batch_size 32 --tanhalpha 2                  4.3480

# python BJ_train.py --layers 3 --var_node_dim 15 --stat_node_dim 15 --batch_size 16 --tanhalpha 2          4.4746

# python BJ_train.py --layers 3 --var_node_dim 20 --stat_node_dim 20 --batch_size 16 --tanhalpha 4            4.3847
# python BJ_train.py --layers 3 --var_node_dim 20 --stat_node_dim 20 --batch_size 16 --tanhalpha 5            4.3934
# python BJ_train.py --layers 3 --var_node_dim 20 --stat_node_dim 20 --batch_size 16 --tanhalpha 6            4.4138


# python BJ_train.py --layers 3 --var_node_dim 40 --stat_node_dim 40 --batch_size 32 --tanhalpha 2         4.3657 
# python BJ_train.py --layers 3 --var_node_dim 40 --stat_node_dim 40 --batch_size 16 --tanhalpha 2            4.4218
# python BJ_train.py --layers 3 --var_node_dim 40 --stat_node_dim 40 --batch_size 16 --tanhalpha 3              4.3480

# python BJ_train.py --layers 2 --var_node_dim 10 --stat_node_dim 10 --batch_size 32
# python BJ_train.py --layers 2 --var_node_dim 10 --stat_node_dim 10 --batch_size 16
# python BJ_train.py --layers 2 --var_node_dim 10 --stat_node_dim 10 --batch_size 16 --tanhalpha 2
# python BJ_train.py --layers 2 --var_node_dim 10 --stat_node_dim 10 --batch_size 32 --tanhalpha 2          
# python BJ_train.py --layers 2 --var_node_dim 15 --stat_node_dim 15 --batch_size 32 --tanhalpha 2          4.3944
# python BJ_train.py --layers 3 --var_node_dim 20 --stat_node_dim 20 --batch_size 16 --tanhalpha 2          4.3331
# python BJ_train.py --layers 3 --var_node_dim 30 --stat_node_dim 30 --batch_size 16 --tanhalpha 2          4.4460
# python BJ_train.py --layers 3 --var_node_dim 10 --stat_node_dim 20 --batch_size 16 --tanhalpha 2          4.3914
# test propalpha
# python BJ_train.py --layers 3 --var_node_dim 5 --stat_node_dim 5 --batch_size 64 --tanhalpha 3              4.4444
# python BJ_train.py --layers 3 --var_node_dim 20 --stat_node_dim 20 --batch_size 64 --tanhalpha 3 --propalpha 0.01   4.3572
# python BJ_train.py --layers 3 --var_node_dim 20 --stat_node_dim 20 --batch_size 64 --tanhalpha 3 --propalpha 0.1    4.3229
# python BJ_train.py --layers 3 --var_node_dim 10 --stat_node_dim 20 --batch_size 64 --tanhalpha 3 --propalpha 0.1    4.3603
# python BJ_train.py --layers 3 --var_node_dim 20 --stat_node_dim 20 --batch_size 64 --tanhalpha 3 --gcn_depth 1  4.3603
# python BJ_train.py --layers 3 --var_node_dim 20 --stat_node_dim 20 --batch_size 64 --tanhalpha 3 --gcn_depth 3
python main.py --epochs 100 --layers 3 --data data/wfd_BJ --data_name BJ --save ./save/BJ/ --gcn_depth 2 --num_var 9 --var_nodes 9 --stat_nodes 10 --var_node_dim 20 --stat_node_dim 20 --seq_in_len 28 --seq_out_len 33 --batch_size 64 --tanhalpha 3