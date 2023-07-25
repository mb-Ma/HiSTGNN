# python USA_train.py --layers 2 --var_node_dim 5 --stat_node_dim 10 --batch_size 16
# python USA_train.py --layers 2 --var_node_dim 5 --stat_node_dim 20
# python USA_train.py --layers 2 --var_node_dim 20 --stat_node_dim 20
# python USA_train.py --layers 2 --var_node_dim 40 --stat_node_dim 40 --batch_size 16
# python USA_train.py --layers 2 --var_node_dim 40 --stat_node_dim 40 --batch_size 64
python main.py --device cuda:1 --epochs 100 --layers 2 --data data/wfd_USA --data_name USA --save ./save/USA/ --gcn_depth 2 --num_var 4 --var_nodes 4 --stat_nodes 13 --var_node_dim 5 --stat_node_dim 10 --seq_in_len 48 --seq_out_len 24 --batch_size 16 --tanhalpha 2 --patient 5
