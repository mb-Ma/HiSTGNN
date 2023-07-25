import argparse
import time
from util import *
from trainer import DoubleTrainer
from HierNet import HierarchicalNet
from datetime import datetime


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')


def log_info(file, info):
    file.write(info+"\n")

parser = argparse.ArgumentParser()

parser.add_argument('--device',type=str,default='cuda:0',help='')
parser.add_argument('--data',type=str,default='../Weather2K/weather2k_dict.h5',help='data path')
parser.add_argument('--data_name',type=str,default='2k',help='dataset name BJ, Israel, USA, 2k')
parser.add_argument('--save',type=str,default='./save/2k/',help='save path')

parser.add_argument('--hier_true', type=str_to_bool, default=True, help='weather use flat graph')
parser.add_argument('--DIL_true', type=str_to_bool, default=False, help='weather use Dynamic Interaction learning')
parser.add_argument('--gcn_true', type=str_to_bool, default=True, help='whether to add graph convolution layer')
parser.add_argument('--gat_true', type=str_to_bool, default=False, help='whether to add graph attention layer')
parser.add_argument('--buildA_true', type=str_to_bool, default=True,help='whether to construct adaptive adjacency matrix')
parser.add_argument('--scale', type=str, default='max-min',help='{max-min or std}')

parser.add_argument('--gcn_depth',type=int,default=2,help='graph convolution depth')
parser.add_argument('--num_var', type=int, default=8,help='3,4,6,8,9, var num during encoding phrase for ablation study')
parser.add_argument('--var_nodes',type=int,default=8,help='number of nodes/variables')
parser.add_argument('--stat_nodes',type=int,default=200,help='number of nodes/variables')
parser.add_argument('--num_heads',type=int,default=4,help='number of multi-head for graph attention')
parser.add_argument('--dropout',type=float,default=0.3,help='dropout rate')
parser.add_argument('--var_node_dim',type=int,default=20,help='dim of nodes') # adjusted
parser.add_argument('--stat_node_dim',type=int,default=64,help='dim of nodes') # adjusted
parser.add_argument('--dilation_exponential',type=int,default=1,help='dilation exponential')

parser.add_argument('--conv_channels',type=int,default=32,help='convolution channels')
parser.add_argument('--residual_channels',type=int,default=32,help='residual channels')
parser.add_argument('--skip_channels',type=int,default=64,help='skip channels')
parser.add_argument('--end_channels',type=int,default=128,help='end channels')

parser.add_argument('--in_dim',type=int,default=2,help='inputs dimension')
parser.add_argument('--seq_in_len',type=int,default=12,help='input sequence length')
parser.add_argument('--seq_out_len',type=int,default=12,help='output sequence length')
parser.add_argument('--embed_true', type=str_to_bool, default=False, help='whether to do station embedding')

parser.add_argument('--layers',type=int,default=3,help='number of layers')
parser.add_argument('--batch_size',type=int,default=32,help='batch size')
parser.add_argument('--learning_rate',type=float,default=0.001,help='learning rate')
parser.add_argument('--propalpha',type=float,default=0.05,help='prop alpha')
parser.add_argument('--tanhalpha',type=float,default=3,help='adj alpha')
parser.add_argument('--weight_decay',type=float,default=0.0001,help='weight decay rate')
parser.add_argument('--clip',type=int,default=5,help='clip')


parser.add_argument('--step_size1',type=int,default=1,help='step_size')
parser.add_argument('--step_size2',type=int,default=100,help='step_size')

parser.add_argument('--epochs',type=int,default=100,help='num of train epoch')
parser.add_argument('--print_every',type=int,default=20,help='print information every n iteration')
parser.add_argument('--seed',type=int,default=101,help='random seed')
parser.add_argument('--patient',type=int,default=15,help='early stop')

parser.add_argument('--expid',type=int,default=1,help='experiment id')
parser.add_argument('--runs',type=int,default=1,help='number of runs')

args = parser.parse_args()
torch.set_num_threads(3)

                                     
def main(runid):
    current_time = datetime.now().strftime("%Y%m%d%H%M%S")
    log_dir = os.path.join(args.save, current_time)

    if not os.path.exists(log_dir):
        # if not
        os.mkdir(log_dir)

    result_data = log_dir + '/HiSTGNN_HG_{}_DIL_{}_PreA_{}_{}_results.npy'.format(args.hier_true, args.DIL_true,
                                                                                args.buildA_true, args.data_name)

    print("The results save in {}".format(log_dir))
    log_file = open(os.path.join(log_dir, 'log.txt'), "w")

    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    # load data
    device = torch.device(args.device)
    dataloader = load_dataset_all(args.data, args.batch_size, args.batch_size, args.batch_size, args.scale, data_name=args.data_name, GPU=args.device)
    scaler = dataloader['scaler']

    if args.data_name == "BJ":
        conv_k_size = (1, 9, 1)
    elif args.data_name == "2k":
        conv_k_size = (1, 6, 1)
    else:
        conv_k_size = (1, 1, 1)

    model = HierarchicalNet(seq_length=args.seq_in_len, n_var=args.var_nodes, n_stat=args.stat_nodes, var_dim=args.var_node_dim,
                            stat_dim=args.stat_node_dim, device=args.device, tanhalpha=args.tanhalpha, conv_channels=args.conv_channels,
                            gcn_depth=args.gcn_depth, residual_channels=args.residual_channels, in_dim=args.in_dim,
                            dropout=args.dropout, end_channels=args.end_channels, out_dim=args.seq_out_len,
                            propalpha=args.propalpha, predefined_A=args.buildA_true, static_feat=None, dilation_exponential=args.dilation_exponential,
                            layers=args.layers, layer_norm_affline=True,skip_channels=args.skip_channels,
                             gcn_true=args.gcn_true, gat_true=args.gat_true, hier_true=args.hier_true, DIL_true=args.DIL_true, conv_k_size=conv_k_size)
    
    print(args)
    print('The recpetive field size is', model.receptive_field)
    nParams = sum([p.nelement() for p in model.parameters()])
    print('Number of model parameters is', nParams)
    log_info(log_file, str(vars(args)))

    engine = DoubleTrainer(model, args.learning_rate, args.weight_decay, args.clip, args.step_size1, args.seq_out_len, scaler, device)

    print("start training...",flush=True)
    his_train_loss = []
    his_valid_loss =[]
    val_time = []
    train_time = []
    minl = 1e5
    early_stop_cnt = 0

    for i in range(1,args.epochs+1):# epochs
        train_loss = []
        train_mae = []
        train_mape = []
        train_rmse = []
        t1 = time.time()
        dataloader['train_loader'].shuffle() # shuffle train_loader data
        for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
            trainx = torch.Tensor(x).to(device)
            trainx = trainx.transpose(1, 4)
            trainy = torch.Tensor(y).to(device)
            metrics = engine.train(trainx, trainy, iter, data_name=args.data_name)
            train_loss.append(metrics[0])
            train_mae.append(metrics[1])
            train_mape.append(metrics[2])
            train_rmse.append(metrics[3])

            if iter % args.print_every == 0 :
                log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAE: {:.4f}Train MAPE: {:.4f}, Train RMSE: {:.4f}'
                print(log.format(iter, train_loss[-1], train_mae[-1], train_mape[-1], train_rmse[-1]),flush=True)
                log_info(log_file, log.format(iter, train_loss[-1], train_mae[-1], train_mape[-1], train_rmse[-1]))

        t2 = time.time()
        train_time.append(t2-t1)

        #validation
        valid_mae = []
        valid_mape = []
        valid_rmse = []

        s1 = time.time()
        for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
            valx = torch.Tensor(x).to(device)
            valx = valx.transpose(1, 4)
            valy = torch.Tensor(y).to(device)
            metrics = engine.eval(valx, valy, data_name=args.data_name) # first reduce, then expand y:(batch, variables, time)
            valid_mae.append(metrics[0])
            valid_mape.append(metrics[1])
            valid_rmse.append(metrics[2])
        s2 = time.time()
        log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
        print(log.format(i,(s2-s1)))
        val_time.append(s2-s1)

        mtrain_loss = np.mean(train_loss)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_rmse)
        his_train_loss.append(mtrain_loss)

        mvalid_loss = np.mean(valid_mae)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)
        his_valid_loss.append(mvalid_loss)

        log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, ' \
              'Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
        print(log.format(i, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mape, mvalid_rmse, (t2 - t1)), flush=True)
        log_info(log_file, log.format(i, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mape, mvalid_rmse, (t2 - t1)))


        if mvalid_loss<minl:
            torch.save(engine.model.state_dict(), log_dir + "/exp" + str(args.expid) + "_" + str(runid) +".pth")
            minl = mvalid_loss
            early_stop_cnt = 0
        else:
            if early_stop_cnt > args.patient:
                log = 'Training at Epoch: {:03d} ending'.format(early_stop_cnt)
                print(log)
                break
            early_stop_cnt += 1
    
    his_train_loss = np.array(his_train_loss)
    his_valid_loss = np.array(his_valid_loss)
    np.save(os.path.join(log_dir, 'loss_log.npy'), np.array([his_train_loss, his_valid_loss]))

    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

    bestid = np.argmin(his_valid_loss) # get minimal loss
    engine.model.load_state_dict(torch.load(log_dir + "/exp" + str(args.expid) + "_" + str(runid) +".pth"))
    print("Training finished")
    print("The valid loss on best model is", str(round(his_valid_loss[bestid], 4)))


    #test data
    outputs = []
    realy = torch.Tensor(dataloader['y_test']).to(device)
    # realy = realy [:, 4:, :, :, :] # (samples, features, variable, time)
    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        # x = x[:, :-9, :, :, :]
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1, 4)
        with torch.no_grad():
            preds = engine.model(testx)
            if args.data_name == '2k':
                preds = scaler.inverse_transform3(preds)  # (batch, 1, variables, time)
            else:
                preds = scaler.inverse_transform2(preds)
        outputs.append(preds)
    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[:realy.size(0), ...] # (batch, 1, variables, time)

    test_mae, test_mape, test_rmse = metric_avg(yhat, realy)
    log = 'Test in the best epoch over variables: MAE: {:.4f}, MAPE: {:.4f}, RMSE: {:.4f}'
    print(log.format(test_mae, test_mape, test_rmse))
    log_info(log_file, log.format(test_mae, test_mape, test_rmse))
    print('\n\n')

    print('The performance of single variable:')
    for i in range(realy.size(-2)):
        metrics = metrics_ori(yhat[:, :, :, i, :], realy[:, :, :, i, :])
        log = 'Test in the best epoch: MAE: {:.4f}, MAPE: {:.4f}, RMSE: {:.4f}'
        print(log.format(metrics[0], metrics[1], metrics[2]))
        log_info(log_file, log.format(metrics[0], metrics[1], metrics[2]))

    # evaluation step-by-step
    t_mae = []
    t_mape = []
    t_rmse = []
    
    for i in range(args.seq_out_len):
        pred = yhat[:, i:i + 1, :, :, :]  
        real = realy[:, i:i + 1, :, :, :] 
        metrics = metric(pred, real)
        t_mae.append(metrics[0])
        t_mape.append(metrics[1])
        t_rmse.append(metrics[2])
        log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(i + 1, metrics[0], metrics[1], metrics[2]))
        log_info(log_file, log.format(i + 1, metrics[0], metrics[1], metrics[2]))

    print('save predicted result into: ' + result_data)
    np.save(result_data, yhat.cpu())
    print('\n\n')
    log = 'Average Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f} Over {} time steps'
    print(log.format(np.mean(t_mae), np.mean(t_mape), np.mean(t_rmse), args.seq_out_len))
    log_info(log_file, log.format(np.mean(t_mae), np.mean(t_mape), np.mean(t_rmse), args.seq_out_len))
    log_file.close()


def infer(model_dir):
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    # load data
    device = torch.device(args.device)
    dataloader = load_dataset_all(args.data, args.batch_size, args.batch_size, args.batch_size, args.scale,
                                  data_name=args.data_name)
    scaler = dataloader['scaler']

    if args.data_name == "BJ":
        conv_k_size = (1, 7, 1)
    if args.data_name == "2k":
        conv_k_size = (1, 6, 1)
    else:
        conv_k_size = (1, 1, 1)

    model = HierarchicalNet(seq_length=args.seq_in_len, n_var=args.var_nodes, n_stat=args.stat_nodes, var_dim=args.var_node_dim,
                            stat_dim=args.stat_node_dim, device=args.device, tanhalpha=args.tanhalpha, conv_channels=args.conv_channels,
                            gcn_depth=args.gcn_depth, residual_channels=args.residual_channels, in_dim=args.in_dim,
                            dropout=args.dropout, end_channels=args.end_channels, out_dim=args.seq_out_len,
                            propalpha=args.propalpha, predefined_A=args.buildA_true, static_feat=None, dilation_exponential=args.dilation_exponential,
                            layers=args.layers, layer_norm_affline=True,skip_channels=args.skip_channels,
                             gcn_true=args.gcn_true, gat_true=args.gat_true, hier_true=args.hier_true, DIL_true=args.DIL_true, conv_k_size=conv_k_size)

    print(args)
    print('The recpetive field size is', model.receptive_field)
    nParams = sum([p.nelement() for p in model.parameters()])
    print('Number of model parameters is', nParams)

    engine = DoubleTrainer(model, args.learning_rate, args.weight_decay, args.clip, args.step_size1, args.seq_out_len,
                           scaler, device)

    engine.model.load_state_dict(torch.load(model_dir))
    
    outputs = []
    realy = torch.Tensor(dataloader['y_test']).to(device)
    # realy = realy [:, 4:, :, :, :] # (samples, features, variable, time)
    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        # x = x[:, :-9, :, :, :]
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1, 4)
        with torch.no_grad():
            preds = engine.model(testx)
            if args.data_name == '2k':
                preds = scaler.inverse_transform3(preds)  # (batch, 1, variables, time)
            else:
                preds = scaler.inverse_transform2(preds)
        outputs.append(preds)
    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[:realy.size(0), ...] # (batch, 1, variables, time)

    test_mae, test_mape, test_rmse = metric_avg(yhat, realy)
    log = 'Test in the best epoch over variables: MAE: {:.4f}, MAPE: {:.4f}, RMSE: {:.4f}'
    print(log.format(test_mae, test_mape, test_rmse))
    print('\n\n')

    print('The performance of single variable:')
    for i in range(realy.size(-2)):
        metrics = metrics_ori(yhat[:, :, :, i, :], realy[:, :, :, i, :])
        log = 'Test in the best epoch: MAE: {:.4f}, MAPE: {:.4f}, RMSE: {:.4f}'
        print(log.format(metrics[0], metrics[1], metrics[2]))


    # evaluation step-by-step
    t_mae = []
    t_mape = []
    t_rmse = []
    for i in range(args.seq_out_len):
        pred = yhat[:, i:i + 1, :, :, :]  
        real = realy[:, i:i + 1, :, :, :] 
        metrics = metric(pred, real)
        t_mae.append(metrics[0])
        t_mape.append(metrics[1])
        t_rmse.append(metrics[2])
        log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(i + 1, metrics[0], metrics[1], metrics[2]))

    print('\n\n')
    print(log.format(np.mean(t_mae), np.mean(t_mape), np.mean(t_rmse), args.seq_out_len))
    


main(1)
# model_dir = ''
# infer()