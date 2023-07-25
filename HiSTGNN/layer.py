from __future__ import division
import torch
import torch.nn as nn
from torch.nn import init
import numbers
import torch.nn.functional as F


class gat(nn.Module):
    '''
    h'_i = \sigma{\sum_{v_j \in N_{v_i}} \alpha_{ij}*W*h_j}
    '''
    def __init__(self, in_dim, out_dim, num_heads, concat=True, activation=nn.ELU(), dropout_prob=0.6,
                 add_skip_connection=True, bias=True, log_attention_weights=False):
        super(gat, self).__init__()
        self.num_heads = num_heads
        self.out_dim = out_dim
        self.concat = concat
        self.add_skip_connection = add_skip_connection

        # for multi-head attention independent W matrices
        self.linear_proj = nn.Linear(in_dim, num_heads * out_dim, bias=False)

        # for multi-head target node and source node
        self.scoring_fn_target - nn.Parameter(torch.Tensor(1, num_heads, out_dim))
        self.scoring_fn_source - nn.Parameter(torch.Tensor(1, num_heads, out_dim))

        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(num_heads * out_dim))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(out_dim))
        else:
            self.register_parameter('bias', None)

        if add_skip_connection:
            self.skip_proj = nn.Linear(in_dim, num_heads * out_dim, bias=False)
        else:
            self.register_parameter('skip_proj', None)

        self.leakyReLU = nn.LeakyReLU(0.2)
        self.softmax = nn.Softmax(dim=-1)
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout_prob)

        self.log_attention_weights = log_attention_weights
        self.attention_weights = None

        self.init_params()

    def init_params(self):
        nn.init.xavier_normal_(self.linear_proj)
        nn.init.xavier_normal_(self.scoring_fn_target)
        nn.init.xavier_normal_(self.scoring_fn_source)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def skip_concat_bias(self, attention_coefficients, in_nodes_features, out_nodes_features):
        if self.log_attention_weights:
            self.attention_weights = attention_coefficients

        if not out_nodes_features.is_contiguous():
            out_nodes_features = out_nodes_features.contiguous()

        if self.add_skip_connection:
            if out_nodes_features.shape[-1] == in_nodes_features.shape[-1]:
                out_nodes_features += in_nodes_features.unsqueeze(1)
            else:
                out_nodes_features += self.skip_proj(in_nodes_features).view(-1, self.num_heads, self.out_dim)
        if self.concat:
            out_nodes_features = out_nodes_features.view(-1, self.num_heads * self.out_dim)
        else:
            out_nodes_features = out_nodes_features.mean(dim=self.num_heads)

        if self.bias is not None:
            out_nodes_features += self.bias

        return out_nodes_features if self.activation is None else self.activation(out_nodes_features)


class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncwl,vw->ncvl',(x,A))
        return x.contiguous()


class nconv1(nn.Module):
    def __init__(self):
        super(nconv1,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncwl,nvw->ncvl',(x,A))
        return x.contiguous()


class dy_nconv(nn.Module):
    def __init__(self):
        super(dy_nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncvl,nvwl->ncwl',(x,A))
        return x.contiguous()


class linear(nn.Module):
    def __init__(self,c_in,c_out,bias=True):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=bias)

    def forward(self,x):
        return self.mlp(x)


class prop(nn.Module):
    def __init__(self,c_in,c_out,gdep,dropout,alpha):
        super(prop, self).__init__()
        self.nconv = nconv()
        self.mlp = linear(c_in,c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha

    def forward(self,x,adj):
        adj = adj + torch.eye(adj.size(0)).to(x.device)
        d = adj.sum(1)
        h = x
        dv = d
        a = adj / dv.view(-1, 1)
        for i in range(self.gdep):
            h = self.alpha*x + (1-self.alpha)*self.nconv(h,a)
        ho = self.mlp(h)
        return ho


class mixprop(nn.Module):
    '''
    this a rnn architecture, can modified by GRU
    '''
    def __init__(self,c_in,c_out,gdep,dropout,alpha,predA=True):
        super(mixprop, self).__init__()
        if not predA:
            self.nconv = nconv1()
        else:
            self.nconv = nconv()
        self.mlp = linear((gdep+1)*c_in,c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha
        # self.alpha = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)

    def forward(self,x,adj):
        adj = adj + torch.eye(adj.size(0)).to(x.device) # A+I
        d = adj.sum(1) # sum of row axis. i.e. degree -> vector not matrix
        h = x
        out = [h]
        a = adj / d.view(-1, 1) # d^-1(a+i)
        for i in range(self.gdep):
            # beta*X + (1-beta)\hat{A}*H
            h = self.alpha*x + (1-self.alpha)*self.nconv(h,a)
            out.append(h)
        ho = torch.cat(out,dim=1)
        ho = self.mlp(ho)
        return ho


class mixpropA(nn.Module):
    '''
    this a rnn architecture, can modified by GRU
    '''
    def __init__(self,c_in,c_out,gdep,dropout,alpha,predA=True):
        super(mixpropA, self).__init__()
        if not predA:
            self.nconv = nconv1()
        else:
            self.nconv = nconv()
        self.mlp = linear((gdep+1)*c_in,c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha
        # self.alpha = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)

    def forward(self,x,adj):
        h = x
        out = [h]
        a_list = []
        adj = F.relu(adj)
        for i in range(adj.shape[0]):
            _adj = adj[i] + torch.eye(adj[i].size(0)).to(x.device) # A+I
            d = _adj.sum(1) # sum of row axis. i.e. degree -> vector not matrix
            a = _adj / d.view(-1, 1) # d^-1(a+i)
            a_list.append(a)
        c = torch.stack(a_list, 0).to(x.device)
        for i in range(self.gdep):
            # beta*X + (1-beta)\hat{A}*H
            h = self.alpha*x + (1-self.alpha)*self.nconv(h,c)
            out.append(h)
        ho = torch.cat(out,dim=1)
        ho = self.mlp(ho)
        return ho


class mixprop_gat(nn.Module):
    def __init__(self, c_in, c_out, gdep, dropout, alpha, leak_alpha, nheads, num_nodes):
        super(mixprop_gat, self).__init__()
        self.gat = GAT(c_in, c_out, dropout, leak_alpha, nheads, num_nodes)
        self.mlp = linear((gdep + 1) * c_in, c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha

    def forward(self, x, adj):
        adj = adj + torch.eye(adj.size(0)).to(x.device)  # A+I
        h = x
        out = [h]
        for i in range(self.gdep):
            h = self.alpha * x + (1 - self.alpha) * self.gat(h, adj)
            out.append(h)
        ho = torch.cat(out, dim=1)
        ho = self.mlp(ho)
        return ho


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, dropout, alpha, nheads, nnodes):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.attentions = [GraphAttentionLayer(nfeat, nhid, nnodes, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.mlp = linear(nheads*nhid, nhid)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1) # 注意力机制层 只有一层
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.mlp(x)
        return x


# class HeterGRaphConv(nn.Module):
#     def __init__(self, mods, aggregate='sum'):
#         super(HeterGRaphConv, self).__init__()
#         self.mods = nn.ModuleDict(mods)
#         if isinstance(aggregate, str):
#             self.agg_fn = get_aggregate_fn(aggregate)
#         else:
#             self.agg_fn = aggregate

class dy_mixprop(nn.Module):
    def __init__(self,c_in,c_out,gdep,dropout,alpha):
        super(dy_mixprop, self).__init__()
        self.nconv = dy_nconv()
        self.mlp1 = linear((gdep+1)*c_in,c_out)
        self.mlp2 = linear((gdep+1)*c_in,c_out)

        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha
        self.lin1 = linear(c_in,c_in)
        self.lin2 = linear(c_in,c_in)

    def forward(self,x):
        #adj = adj + torch.eye(adj.size(0)).to(x.device)
        #d = adj.sum(1)
        x1 = torch.tanh(self.lin1(x))
        x2 = torch.tanh(self.lin2(x))
        adj = self.nconv(x1.transpose(2,1),x2)
        adj0 = torch.softmax(adj, dim=2)
        adj1 = torch.softmax(adj.transpose(2,1), dim=2)

        h = x
        out = [h]
        for i in range(self.gdep):
            h = self.alpha*x + (1-self.alpha)*self.nconv(h,adj0)
            out.append(h)
        ho = torch.cat(out,dim=1)
        ho1 = self.mlp1(ho)

        h = x
        out = [h]
        for i in range(self.gdep):
            h = self.alpha * x + (1 - self.alpha) * self.nconv(h, adj1)
            out.append(h)
        ho = torch.cat(out, dim=1)
        ho2 = self.mlp2(ho)

        return ho1+ho2


class dilated_1D(nn.Module):
    def __init__(self, cin, cout, dilation_factor=2):
        super(dilated_1D, self).__init__()
        self.tconv = nn.ModuleList()
        self.kernel_set = [2,3,6,7]
        self.tconv = nn.Conv2d(cin,cout,(1,7),dilation=(1,dilation_factor))

    def forward(self,input):
        x = self.tconv(input)
        return x


class dilated_inception(nn.Module):
    def __init__(self, cin, cout, dilation_factor=2):
        super(dilated_inception, self).__init__()
        self.tconv = nn.ModuleList()
        # self.kernel_set = [2, 3, 6, 7]
        self.kernel_set = [3, 6]
        cout = int(cout/len(self.kernel_set))
        for kern in self.kernel_set:
            self.tconv.append(nn.Conv2d(cin,cout,(1,kern),dilation=(1, dilation_factor)))

    def forward(self,input):
        x = []
        for i in range(len(self.kernel_set)):
            x.append(self.tconv[i](input))
        for i in range(len(self.kernel_set)):
            x[i] = x[i][...,-x[-1].size(3):]
        x = torch.cat(x, dim=1)
        return x


class conv_6(nn.Module):
    def __init__(self, cin, cout, dilation_factor=2):
        super(conv_6, self).__init__()
        self.conv_2 = nn.Conv2d(cin, cout, (1, 2), dilation=(1, dilation_factor))
        self.conv_3 = nn.Conv2d(cout, cout, (1, 3), dilation=(1, dilation_factor))

    def forward(self, input):
        return self.conv_3(self.conv_2(input))


class conv_7(nn.Module):
    def __init__(self, cin, cout, dilation_factor=2):
        super(conv_7, self).__init__()
        self.conv_3_1 = nn.Conv2d(cin, cout, (1, 3), dilation=(1, dilation_factor))
        self.conv_3_2 = nn.Conv2d(cout, cout//2, (1, 3), dilation=(1, dilation_factor))
        self.conv_3_3 = nn.Conv2d(cout//2, cout, (1, 3), dilation=(1, dilation_factor))

    def forward(self, input):
        return self.conv_3_3(self.conv_3_2(self.conv_3_1(input)))


class dilated_inception_v2(nn.Module):
    def __init__(self, cin, cout, dilation_factor=2):
        super(dilated_inception_v2, self).__init__()
        self.tconv = nn.ModuleList()
        cout = int(cout/4)
        self.conv_2 = nn.Conv2d(cin, cout, (1, 2), dilation=(1, dilation_factor))
        self.conv_3 = nn.Conv2d(cin, cout, (1, 3), dilation=(1, dilation_factor))
        self.conv_6 = conv_6(cin, cout, dilation_factor=dilation_factor)
        self.conv_7 = conv_7(cin, cout, dilation_factor=dilation_factor)

        self.tconv.append(self.conv_2)
        self.tconv.append(self.conv_3)
        self.tconv.append(self.conv_6)
        self.tconv.append(self.conv_7)

    def forward(self,input):
        x = []
        for i in range(4):
            x.append(self.tconv[i](input))
        for i in range(4):
            x[i] = x[i][...,-x[-1].size(3):]
        x = torch.cat(x, dim=1)
        return x


class dilated_inception_same(nn.Module):
    def __init__(self, cin, cout, dilation_factor=2):
        super(dilated_inception_same, self).__init__()
        self.tconv = nn.ModuleList()
        # self.kernel_set = [2, 3, 6, 7]
        self.kernel_set = [3, 6]
        cout = int(cout/len(self.kernel_set))
        for kern in self.kernel_set:
            self.tconv.append(nn.Conv2d(cin,cout,(1,kern),dilation=(1, dilation_factor), padding=(0, 3)))

    def forward(self,input):
        x = []
        for i in range(len(self.kernel_set)):
            x.append(self.tconv[i](input))
        for i in range(len(self.kernel_set)):
            x[i] = x[i][...,-input.size(3):]
        x = torch.cat(x, dim=1)
        return x


class dilated_3D(nn.Module):
    def __init__(self, cin, cout, dilation_factor=2):
        super(dilated_3D, self).__init__()
        self.tconv = nn.Conv3d(cin,cout,(1, 1, 3),dilation=(dilation_factor, 1, 1))

    def forward(self,input):
        x = self.tconv(input)
        return x


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


class graph_global(nn.Module):
    def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None):
        super(graph_global, self).__init__()
        self.nnodes = nnodes
        self.A = nn.Parameter(torch.randn(nnodes, nnodes).to(device), requires_grad=True).to(device)

    def forward(self, idx):
        return F.relu(self.A)


class graph_undirected(nn.Module):
    def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None):
        super(graph_undirected, self).__init__()
        self.nnodes = nnodes
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(nnodes, dim)
            self.lin1 = nn.Linear(dim,dim)

        self.device = device
        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat

    def forward(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb1(idx)
        else:
            nodevec1 = self.static_feat[idx,:]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha*self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha*self.lin1(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1,0))
        adj = F.relu(torch.tanh(self.alpha*a))
        mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device)
        mask.fill_(float('0'))
        s1,t1 = adj.topk(self.k,1)
        mask.scatter_(1,t1,s1.fill_(1))
        adj = adj*mask
        return adj


class graph_directed(nn.Module):
    def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None):
        super(graph_directed, self).__init__()
        self.nnodes = nnodes
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
            self.lin2 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(nnodes, dim)
            self.emb2 = nn.Embedding(nnodes, dim)
            self.lin1 = nn.Linear(dim,dim)
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

        nodevec1 = torch.tanh(self.alpha*self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha*self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1,0))
        adj = F.relu(torch.tanh(self.alpha*a))
        mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device)
        mask.fill_(float('0'))
        s1,t1 = adj.topk(self.k,1)
        mask.scatter_(1,t1,s1.fill_(1))
        adj = adj*mask
        return adj


class LayerNorm(nn.Module):
    __constants__ = ['normalized_shape', 'weight', 'bias', 'eps', 'elementwise_affine']

    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.Tensor(*normalized_shape))
            self.bias = nn.Parameter(torch.Tensor(*normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input, idx):
        if self.elementwise_affine:
            return F.layer_norm(input, tuple(input.shape[1:]), self.weight[:,idx,:], self.bias[:,idx,:], self.eps)
        else:
            return F.layer_norm(input, tuple(input.shape[1:]), self.weight, self.bias, self.eps)

    def extra_repr(self):
        return '{normalized_shape}, eps={eps}, ' \
            'elementwise_affine={elementwise_affine}'.format(**self.__dict__)
