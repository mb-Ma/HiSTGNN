from layer import *
import numpy as np

'''
input dimension [batch, time, stations, variables, features]
output dimension [batch, time, stations, variables(only 3), 1]

'''

class graph_constructor(nn.Module):
    '''
    Graph Learning Layer
    nnodes: num of nodes
    k: subgraph_size
    dim: node dimension default 40
    return adjacency matrix is learned from data
    '''
    def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None):
        super(graph_constructor, self).__init__()
        self.nnodes = nnodes
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
            self.lin2 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(nnodes, dim) # node embedding
            self.emb2 = nn.Embedding(nnodes, dim) # node embedding
            self.lin1 = nn.Linear(dim,dim) #
            self.lin2 = nn.Linear(dim,dim)

        self.device = device
        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat

    def forward(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx,:]
            nodevec2 = nodevec1

        # 无向关系
        # nodevec1 = torch.tanh(self.alpha * self.lin1(nodevec1))  # equation 1
        # a = torch.mm(nodevec1, nodevec1.transpose(1,0))
        nodevec1 = torch.tanh(self.alpha*self.lin1(nodevec1)) # equation 1
        nodevec2 = torch.tanh(self.alpha*self.lin2(nodevec2)) # equation 2

        a = torch.mm(nodevec1, nodevec2.transpose(1,0))-torch.mm(nodevec2, nodevec1.transpose(1,0)) # equation3
        adj = F.relu(torch.tanh(self.alpha*a)) # equation 3

        mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device) # 获得第0维 构造一个矩阵
        mask.fill_(float('0')) # character 0 fill

        s1, t1 = adj.topk(self.k,1) # s1: values t1: indices
        mask.scatter_(1,t1,s1.fill_(1)) # 被１填充，以保证在进行乘法运算后邻接非０值保持不变
        adj = adj*mask # 生成邻接矩阵
        return adj

    def fullA(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx,:]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha*self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha*self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1,0))-torch.mm(nodevec2, nodevec1.transpose(1,0))
        adj = F.relu(torch.tanh(self.alpha*a))
        return adj


class graph_constructor_directed(nn.Module):
    '''
    Graph Learning Layer
    nnodes: num of nodes
    k: subgraph_size
    dim: node dimension default 40
    return adjacency matrix is learned from data
    '''
    def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None):
        super(graph_constructor_directed, self).__init__()
        self.nnodes = nnodes
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
            self.lin2 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(nnodes, dim) # node embedding
            self.emb2 = nn.Embedding(nnodes, dim) # node embedding
            self.lin1 = nn.Linear(dim,dim) #
            self.lin2 = nn.Linear(dim,dim)

        self.device = device
        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat

    def forward(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx,:]
            nodevec2 = nodevec1

        a = torch.mm(nodevec1, nodevec2.transpose(1,0))
        adj = F.relu(a)
        adj = F.softmax(adj)
        return adj


class graph_constructor_undirected(nn.Module):
    '''
    Graph Learning Layer
    nnodes: num of nodes
    k: subgraph_size
    dim: node dimension default 40
    return adjacency matrix is learned from data
    '''
    def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None):
        super(graph_constructor_undirected, self).__init__()
        self.nnodes = nnodes
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(nnodes, dim) # node embedding
            self.lin1 = nn.Linear(dim,dim) #

        self.device = device
        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat

    def forward(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
        else:
            nodevec1 = self.static_feat[idx,:]

        a = torch.mm(nodevec1, nodevec1.transpose(1,0))
        adj = F.relu(a)
        adj = F.softmax(adj)
        return adj



class HierarchicalNet(nn.Module):
    def __init__(self, seq_length, n_var, n_stat, var_dim, stat_dim, device, tanhalpha, conv_channels,gcn_depth,
                 residual_channels=32, in_dim=2, dropout=0.3, end_channels=128, out_dim=33, propalpha=0.05, predefined_A=True,
                 static_feat=None, dilation_exponential=1, layers=1, layer_norm_affline=True,skip_channels=64,
                 gcn_true=False, gat_true=False, num_heads=4, hier_true=True, DIL_true=True, fusion='AvgPool', 
                 diffusion='GatedCopy', A_type='uni-directed', conv_k_size=(1, 1, 1), kernel_size = 6):
        super(HierarchicalNet, self).__init__()
        self.n_var = n_var
        self.n_stat = n_stat
        self.var_dim = var_dim
        self.stat_dim = stat_dim
        self.layers = layers
        self.predefined_A = predefined_A
        self.var_graph = nn.ModuleList()
        self.var_graph_gcn = nn.ModuleList()
        self.var_graph_gat = nn.ModuleList()

        self.convs = nn.ModuleList()
        self.stat_convs = nn.ModuleList()
        self.stat_graph_gcn = nn.ModuleList()
        self.stat_graph_gat = nn.ModuleList()
        self.stat_norm = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.dropout = dropout
        self.norm = nn.ModuleList()
        self.gcn_true = gcn_true
        self.gat_true = gat_true
        self.gate_convs = nn.ModuleList()
        self.stat_gate = nn.ModuleList()
        self.seq_length = seq_length

        self.hier_true = hier_true
        self.DIL_true = DIL_true

        self.fusion = fusion
        self.diffusion = diffusion
        self.A_type = A_type

        # graph structure building
        if not self.predefined_A:
            self.predefined_A_gg = torch.Tensor(self.load_predA('data/wfd_BJ/gg.npy')).to(device)
            self.predefined_A_lg = torch.Tensor(self.load_predA('data/wfd_BJ/lg.npy')).to(device)
        # 1. local graphs
        for i in range(n_stat):
            if self.A_type == 'uni-directed':
                self.var_graph.append(graph_constructor(n_var, n_var, var_dim, device, alpha=tanhalpha, static_feat=static_feat))
            elif self.A_type == 'directed':
                self.var_graph.append(
                    graph_constructor_directed(n_var, n_var, var_dim, device, alpha=tanhalpha, static_feat=static_feat))
            else:
                self.var_graph.append(
                    graph_constructor_undirected(n_var, n_var, var_dim, device, alpha=tanhalpha, static_feat=static_feat))
        # 2. globl graph
        if self.A_type == 'uni-directed':
            self.stat_gc = graph_constructor(n_stat, n_stat, stat_dim, device, alpha=tanhalpha, static_feat=static_feat)
        elif self.A_type == 'directed':
            self.stat_gc = graph_constructor_directed(n_stat, n_stat, stat_dim, device, alpha=tanhalpha, static_feat=static_feat)
        else:
            self.stat_gc = graph_constructor_undirected(n_stat, n_stat, stat_dim, device, alpha=tanhalpha, static_feat=static_feat)

        
        self.start_conv = nn.Conv3d(in_channels=in_dim, out_channels=residual_channels,
                                    kernel_size=(1, 1, 1),
                                    bias=True)
        
        # receptive filed setting
        if dilation_exponential > 1:
            self.receptive_field = int(
                1 + (kernel_size - 1) * (dilation_exponential ** layers - 1) / (dilation_exponential - 1))
        else:
            self.receptive_field = layers * (kernel_size - 1) + 1

        if dilation_exponential > 1:
            rf_size_i = int(1 + 0 * (kernel_size - 1) * (dilation_exponential ** layers - 1) / (dilation_exponential - 1))
        else:
            rf_size_i = 0 * layers * (kernel_size - 1) + 1
        
        new_dilation = 1

        # layers
        for j in range(1, self.layers+1):
            if dilation_exponential > 1:
                rf_size_j = int(
                    rf_size_i + (kernel_size - 1) * (dilation_exponential ** j - 1) / (dilation_exponential - 1))
            else:
                rf_size_j = rf_size_i + j * (kernel_size - 1)

            if self.DIL_true:
                self.stat_convs.append(dilated_inception(residual_channels, conv_channels, dilation_factor=new_dilation))
                self.stat_gate.append(dilated_inception(residual_channels, conv_channels, dilation_factor=new_dilation))
            else:
                self.stat_convs.append(dilated_inception_same(residual_channels, conv_channels, dilation_factor=new_dilation))
                self.stat_gate.append(dilated_inception_same(residual_channels, conv_channels, dilation_factor=new_dilation))

            for k in range(self.n_stat):
                # stations
                self.convs.append(dilated_inception(residual_channels, conv_channels, dilation_factor=new_dilation))
                self.gate_convs.append(dilated_inception(residual_channels, conv_channels, dilation_factor=new_dilation))
                # add one normalization to avoid rh2m variable
                if gcn_true:
                    self.var_graph_gcn.append(mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha))
                if gat_true:
                    self.var_graph_gat.append(mixprop_gat(conv_channels, residual_channels, gcn_depth, dropout, propalpha, 0.2, num_heads, n_var))

                if self.seq_length > self.receptive_field:
                    self.norm.append(LayerNorm((residual_channels, n_var, self.seq_length - rf_size_j + 1),
                                               elementwise_affine=layer_norm_affline))
                else:
                    self.norm.append(LayerNorm((residual_channels, n_var, self.receptive_field - rf_size_j + 1),
                                               elementwise_affine=layer_norm_affline))

            if self.seq_length>self.receptive_field:
                self.skip_convs.append(nn.Conv3d(in_channels=conv_channels,
                                                out_channels=skip_channels,
                                                kernel_size=(1, 1, self.seq_length-rf_size_j+1)))
            else:
                self.skip_convs.append(nn.Conv3d(in_channels=conv_channels,
                                                out_channels=skip_channels,
                                                kernel_size=(1, 1, self.receptive_field-rf_size_j+1)))

            new_dilation *= dilation_exponential
            if gcn_true:
                self.stat_graph_gcn.append(mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha))
            if gat_true:
                self.stat_graph_gat.append(mixprop_gat(conv_channels, residual_channels, gcn_depth, dropout, propalpha, 0.2, num_heads, n_stat))

            if self.seq_length > self.receptive_field:
                self.stat_norm.append(LayerNorm((residual_channels, n_stat, self.seq_length - rf_size_j + 1),
                                                elementwise_affine=layer_norm_affline))
            else:
                self.stat_norm.append(LayerNorm((residual_channels, n_stat, self.receptive_field - rf_size_j + 1),
                                                elementwise_affine=layer_norm_affline))

        self.end_conv_1 = nn.Conv3d(in_channels=skip_channels,
                                             out_channels=end_channels,
                                             kernel_size=(1, 1, 1),
                                             bias=True)

        self.end_conv_2 = nn.Conv3d(in_channels=end_channels,
                                             out_channels=out_dim,
                                             kernel_size=conv_k_size,
                                             bias=True)

        self.var_idx = torch.arange(self.n_var).to(device)
        self.stat_idx = torch.arange(self.n_stat).to(device)

        if self.seq_length > self.receptive_field:
            self.skip0 = nn.Conv3d(in_channels=in_dim, out_channels=skip_channels, kernel_size=(1, 1, self.seq_length), bias=True)
            self.skipE = nn.Conv3d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, 1, self.seq_length-self.receptive_field+1), bias=True)

        else:
            self.skip0 = nn.Conv3d(in_channels=in_dim, out_channels=skip_channels, kernel_size=(1, 1, self.receptive_field), bias=True)
            self.skipE = nn.Conv3d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, 1, 1), bias=True)

        self.bn = nn.BatchNorm3d(in_dim, affine=False)
        self.nwp_start_conv = nn.Conv3d(in_channels=in_dim, out_channels=residual_channels,kernel_size=(1,1,1))

        self.weight1 = torch.nn.Parameter(torch.FloatTensor(10,3,1), requires_grad=True)
        self.weight2 = torch.nn.Parameter(torch.FloatTensor(10,3,1), requires_grad=True)
        self.norm_weight()

        self.nwp_start_conv = torch.nn.Conv3d(in_channels=in_dim, out_channels=residual_channels,
                                            kernel_size=(1, 1, 1), bias=True)
        self.feat_conv = torch.nn.Conv3d(in_channels=residual_channels, out_channels=1,
                                            kernel_size=(1, 27, 1), bias=True)

    def load_predA(self, dir):
        return np.load(dir)

    def norm_weight(self):
        nn.init.xavier_uniform_(self.weight1)
        nn.init.xavier_uniform_(self.weight2)

    def forward(self, input):
        '''
        1. encoder phrase
            1. start_conv -> temporal learning -> spatial learning; (this is for local graph)
            2. local graph -> global graph by graph representation
            3. temporal learning -> spatial learning (this is for global graph)
            4. encapsulate above module to one layer

        2. decoder phrase
            input NWP data, fusion is better way, rather than get local and global representation of NWP.

        3. predictor
        residual connection is necessary
        '''

        # input.shape [batch, feature, stat_nodes, var_nodes, time]
        seq_len = input.size(4)  # input length
        assert seq_len == self.seq_length, 'input sequence length not equal to preset sequence length'
        # if receptive filed bigger than input length, padding is necessary.
        if self.seq_length < self.receptive_field:
            input = nn.functional.pad(input, (self.receptive_field - self.seq_length, 0, 0, 0))  # 做padding

        # variable graph i.e. local graph
        var_adp = []
        for i in range(self.n_stat):
            if not self.predefined_A:
                adp = self.predefined_A_lg[i]
            else:
                adp = self.var_graph[i](self.var_idx)
            var_adp.append(adp)
        if self.hier_true or self.DIL_true:
            # station graph
            if not self.predefined_A:
                stat_adp = self.predefined_A_gg
            else:
                stat_adp = self.stat_gc(self.stat_idx)

        # input = self.bn(input)
        # [32, 32, 10, 9, 28]
        x = self.start_conv(input) 
        skip = self.skip0(F.dropout(input, self.dropout, training=self.training)) # out: skip_channels
        # encoding layer
        # i^{th} layer， j^{th} station
        if self.hier_true:
            for i in range(self.layers):
                residual = x
                stat_list = []
                # local graph temporal learning
                stat_temp_convs = []
                for j in range(self.n_stat):
                    # print("#{} layer, #{} station".format(i,j))
                    # ---------------- Gate Temporal COnvolution  --------------------
                    _x = self.convs[i*self.n_stat+j](x[:, :, j, :, :])
                    _x = torch.tanh(_x)
                    _gx = self.gate_convs[i*self.n_stat+j](x[:, :, j, :, :])
                    _gx = torch.sigmoid(_gx)
                    _x = _x * _gx 

                    # layer normalization
                    _x = self.norm[j + i * self.n_stat](_x, self.var_idx)  
                    _x = F.dropout(_x, self.dropout, training=self.training)
                    stat_temp_convs.append(_x)

                    # -----------------Spatial Graph Convolution --------
                    # per station spatial learning
                    if self.gcn_true:
                        _x = self.var_graph_gcn[j](_x, var_adp[j]) + self.var_graph_gcn[j](_x, var_adp[j].transpose(1,0))  
                    if self.gat_true:
                        _x = self.var_graph_gat[j](_x, var_adp[j]) + self.var_graph_gat[j](_x, var_adp[j].transpose(1,0))
                    # layer normalization
                    _x = _x + residual[:, :, j, :, -_x.size(3):]
                    _x = self.norm[j + i*self.n_stat](_x, self.var_idx)  # layer normalization
                    # calculate single station representation
                    stat_list.append(_x)

                x = torch.stack(stat_temp_convs, dim=2)
                s = x
                s = self.skip_convs[i](s)
                skip = s + skip

                # ----------------- Information Fusion Layer ------------------
                # station process, and this keep_x is different with x. # [B, F, N, M, T]
                keep_x = torch.stack(stat_list, dim=2) 
                # var -> station
                if self.fusion != 'AvgPool':
                    stat_x = F.adaptive_avg_pool3d(keep_x, (self.n_stat, 1, keep_x.size(-1))).squeeze()
                else:
                    stat_x = F.adaptive_max_pool3d(keep_x, (self.n_stat, 1, keep_x.size(-1))).squeeze()
                # stat_x = torch.mean(keep_x, dim=3, keepdim=False)
                if len(stat_x.shape) < 4:
                    stat_x = stat_x.unsqueeze(-1)
                stat_residual = stat_x
                # ----------------- Gate Temporal Convolution for Global level----------
                stat_filter = self.stat_convs[i](stat_x) # [B, F, N, T]
                stat_filter = torch.tanh(stat_filter)
                stat_gate = self.stat_gate[i](stat_x)
                stat_gate = torch.sigmoid(stat_gate)
                stat_x = stat_filter * stat_gate
                # stat_filter = self.bn(stat_filter)
                stat_x = self.stat_norm[i](stat_x, self.stat_idx)
                stat_x = F.dropout(stat_x, self.dropout, training=self.training)  # per station temporal learning
                # ----------------- Spatial Graph Convolution for Global level----------
                # whole station spatial learning
                if self.gcn_true:
                    stat_x = self.stat_graph_gcn[i](stat_x, stat_adp) + self.stat_graph_gcn[i](stat_x, stat_adp.transpose(1, 0))
                if self.gat_true:
                    stat_x = self.stat_graph_gat[i](stat_x, stat_adp) + self.stat_graph_gat[i](stat_x, stat_adp.transpose(1, 0))
                # layer normalization
                stat_x = stat_x + stat_residual[:, :, :, -stat_x.size(3):]
                # layer normalization [B, F, N, T]
                stat_x = self.stat_norm[i](stat_x, self.stat_idx)  

                # ------------- Information Diffusion Layer -----------------------
                # get station-level then transfer coarse grained graph -> fined grained graph [B, F, N, T]->[B, F, N, M, T]

                stat_x = torch.unsqueeze(stat_x, 3).repeat(1, 1, 1, self.n_var, 1)
                if self.diffusion != 'GatedCopy':
                    stat_x_gate = torch.sigmoid(stat_x) # last dimension is not matched
                    stat_x_filter = torch.tanh(stat_x)
                    stat_x = stat_x_gate * stat_x_filter
                x = torch.mul(stat_x, residual[:, :, :, :, -stat_x.size(4):])

        elif self.DIL_true:
            if self.fusion != 'AvgPool':
                stat_x = F.adaptive_avg_pool3d(x, (self.n_stat, 1, x.size(-1))).squeeze()
            else:
                stat_x = F.adaptive_max_pool3d(x, (self.n_stat, 1, x.size(-1))).squeeze()
            for i in range(self.layers):
                residual = x
                stat_list = []
                # local graph temporal learning
                stat_temp_convs = []
                for j in range(self.n_stat):
                    _x = self.convs[i*self.n_stat+j](x[:, :, j, :, :])
                    _x = torch.tanh(_x)
                    _gx = self.gate_convs[i*self.n_stat+j](x[:, :, j, :, :])
                    _gx = torch.sigmoid(_gx)
                    _x = _x * _gx 

                    # _x = self.bn(_x)
                    _x = self.norm[j + i * self.n_stat](_x, self.var_idx)  # layer normalization
                    _x = F.dropout(_x, self.dropout, training=self.training)
                    stat_temp_convs.append(_x)
                    # per station spatial learning
                    if self.gcn_true:
                        _x = self.var_graph_gcn[j](_x, var_adp[j]) + self.var_graph_gcn[j](_x, var_adp[j].transpose(1,0))  # [32, 32, 9, 22]
                    if self.gat_true:
                        _x = self.var_graph_gat[j](_x, var_adp[j]) + self.var_graph_gat[j](_x, var_adp[j].transpose(1,0))  # [32, 32, 9, 22]
                    # layer normalization
                    _x = _x + residual[:, :, j, :, -_x.size(3):]
                    _x = self.norm[j + i*self.n_stat](_x, self.var_idx)  # layer normalization
                    # calculate single station representation
                    stat_list.append(_x)
                x = torch.stack(stat_temp_convs, dim=2)
                s = x
                s = self.skip_convs[i](s)
                skip = s + skip
                x = torch.stack(stat_list, dim=2)

                stat_residual = stat_x
                stat_filter = self.stat_convs[i](stat_x)  # [B, F, N, T]
                stat_filter = torch.tanh(stat_filter)
                stat_gate = self.stat_gate[i](stat_x)
                stat_gate = torch.sigmoid(stat_gate)
                stat_x = stat_filter * stat_gate
                stat_x = self.stat_norm[i](stat_x, self.stat_idx)
                stat_x = F.dropout(stat_x, self.dropout, training=self.training)  # per station temporal learning
                # whole station spatial learning
                if self.gcn_true:
                    stat_x = self.stat_graph_gcn[i](stat_x, stat_adp) + self.stat_graph_gcn[i](stat_x,stat_adp.transpose(1, 0))
                if self.gat_true:
                    stat_x = self.stat_graph_gat[i](stat_x, stat_adp) + self.stat_graph_gat[i](stat_x,stat_adp.transpose(1, 0))
                # layer normalization
                stat_x = stat_x + stat_residual[:, :, :, -stat_x.size(3):]
                stat_x = self.stat_norm[i](stat_x, self.stat_idx)  # layer normalization [B, F, N, T]
            
            stat_x = torch.unsqueeze(stat_x, 3).repeat(1, 1, 1, self.n_var, 1)
            if self.diffusion != 'GatedCopy':
                stat_x_gate = torch.sigmoid(stat_x)  # last dimension is not matched
                stat_x_filter = torch.tanh(stat_x)
                stat_x = stat_x_gate * stat_x_filter
            x = torch.mul(stat_x, residual[:, :, :, :, -stat_x.size(4):])  # stat_x[..., 16], keep_x [..., 22]
        else:
            for i in range(self.layers):
                residual = x
                stat_list = []
                # local graph temporal learning
                stat_temp_convs = []
                for j in range(self.n_stat):
                    _x = self.convs[i*self.n_stat+j](x[:, :, j, :, :])
                    _x = torch.tanh(_x)
                    _gx = self.gate_convs[i*self.n_stat+j](x[:, :, j, :, :])
                    _gx = torch.sigmoid(_gx)
                    _x = _x * _gx  

                    # _x = self.bn(_x)
                    _x = self.norm[j + i * self.n_stat](_x, self.var_idx)  # layer normalization
                    _x = F.dropout(_x, self.dropout, training=self.training)
                    stat_temp_convs.append(_x)
                    # per station spatial learning
                    if self.gcn_true:
                        _x = self.var_graph_gcn[j](_x, var_adp[j]) + self.var_graph_gcn[j](_x, var_adp[j].transpose(1,0))  # [32, 32, 9, 22]
                    if self.gat_true:
                        _x = self.var_graph_gat[j](_x, var_adp[j]) + self.var_graph_gat[j](_x, var_adp[j].transpose(1,0))  # [32, 32, 9, 22]
                    # layer normalization
                    _x = _x + residual[:, :, j, :, -_x.size(3):]
                    _x = self.norm[j + i*self.n_stat](_x, self.var_idx)  # layer normalization
                    # calculate single station representation
                    stat_list.append(_x)
                x = torch.stack(stat_temp_convs, dim=2)
                s = x
                s = self.skip_convs[i](s)
                skip = s + skip
                x = torch.stack(stat_list, dim=2)

        skip = self.skipE(x) + skip
        x = F.relu(skip)
        # input:[32, 64, 10, 9, 1] -> out:[32, 64, 10, 9, 1]
        x = F.relu(self.end_conv_1(x)) 
        # [B, F, N, M, T] -> [B, 33, N, 3, 1] 
        x = self.end_conv_2(x)  
        return x