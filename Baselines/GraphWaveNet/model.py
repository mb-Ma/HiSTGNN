import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np
from torch.autograd import Variable
import sys


class dynconv(nn.Module):
    def __init__(self):
        super(dynconv, self).__init__()

    def forward(self, x, A):
        # A shape bxnxn
        x = torch.einsum('ncvl,nvw->ncwl', x, A)
        return x.contiguous()


class nconv(nn.Module):
    '''
    graph convolution with adjacency
    '''
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        '''

        :param args x: a 4D Tensor, (batch_size, input_dim, num_node, num_time),
        :param args A: a 2D Tensor, (num_node, num_node)
        :return:
        '''
        x = torch.einsum('ncvl,vw->ncwl',(x,A))
        return x.contiguous()


class linear(nn.Module):
    def __init__(self,c_in,c_out):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)

    def forward(self,x):
        return self.mlp(x)


class gcn_2(nn.Module):
    def __init__(self,c_in,c_out,dropout,support_len=3,order=2):
        super(gcn,self).__init__()
        self.nconv = nconv()
        self.dyncony = dynconv()
        c_in = (order*support_len+1)*c_in
        self.mlp = linear(c_in,c_out)
        self.dropout = dropout
        self.order = order

    def forward(self,x,support):
        out = [x]
        # for a in support:
        #     x1 = self.nconv(x,a)
        #     out.append(x1)
        #     for k in range(2, self.order + 1):
        #         x2 = self.nconv(x1,a)
        #         out.append(x2)
        #         x1 = x2
        x1 = self.dyncony(x, support[0])
        out.append(x1)
        out.append(self.dyncony(x1, support[0]))
        x1 = self.nconv(x, support[1])
        out.append(x1)
        out.append(self.nconv(x1, support[1]))

        h = torch.cat(out,dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class gcn(nn.Module):
    def __init__(self,c_in,c_out,dropout,support_len=3,order=3):
        '''
        :param c_in: input dim
        :param c_out: output dim
        :param support_len: the num of adj
        :param order: diffusion steps
        '''
        super(gcn,self).__init__()
        self.nconv = nconv()
        c_in = (order*support_len+1)*c_in # plus the x.
        self.mlp = linear(c_in,c_out)
        self.dropout = dropout
        self.order = order

    def forward(self,x,support):
        out = [x]
        for a in support: # each relational edge (A)
            x1 = self.nconv(x,a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1,a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out,dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class hgcn(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        '''
        :param c_in: input dim
        :param c_out: output dim
        :param support_len: the num of adj
        :param order: diffusion steps
        '''
        super(hgcn, self).__init__()
        self.nconv = nconv()
        self.rel_in = order * c_in
        self.rel_out = c_in
        c_in = (support_len + 1) * c_in
        self.mlp = linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

        # Rel weights
        self.W_r = Parameter(torch.FloatTensor(support_len, self.rel_in, self.rel_out))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_r)

    def forward(self, x, supports):
        '''
        There are two methods to model multi-relational heterogeneous graph.
        1. first do graph convolution with order_num diffusion for one relation, then concat all their output.
        'CONCAT(X, XA_0, XA_1,..., XA_n, XA_0^2, XA_1^2, ..., XA_n^2)W'
        2. CONCAT(X, CONCAT(XA_0, XA_0^2,..., XA_0^{Order})W_{r0}, CONCAT(XA_1, XA_1^2,..., XA_1^{Order})W_{r1})W
        '''
        out = []
        for a in supports:  # each relational edge (A)
            x1 = self.nconv(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2
        rel_out = [] # rel_num, batch_size, rel_in, num_node, T
        for i in range(len(supports)):
            rel_out.append(torch.cat(out[i * self.order:(i+1)*self.order], dim=1))
        rel_out = torch.stack(rel_out, dim=1) # [batch_size, rel_num, rel_in, num_node, T]

        # do aggregate for each relation [rel_num, rel_in, rel_out]
        rel_out = torch.einsum('brvnt, rvj->brjnt', (rel_out, self.W_r))
        # h = rel_out.view(*rel_out.shape[:1], -1, *rel_out.shape[4:]).contiguous()
        h = torch.reshape(rel_out, (*rel_out.shape[:1], -1, *rel_out.shape[3:])) # [batch_size, c_in, num_node, T]
        h = torch.cat([h, x], dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class Big2SmallConv2d(nn.Module):
    """
    replace big size kernel with  small size kernel on dilation convolution
    1. first calculate the output dim based on given kernel size and dilated rate.
    2. fix the #base_kernel_size=16, the #layers=4, adjust kernel_size based on given base dilated rate.
    """
    def __init__(self, c_in, c_out, base_kernel_size=16, kernel_size=5, base_dilated_rate=1):
        super(Big2SmallConv2d, self).__init__()
        # following formula is the initial solution
        # reduced_size = base_dilated_rate * (base_kernel_size - 1)
        # self.num_layers = int(np.log2(reduced_size // (kernel_size - 1) + 1))

        # current solution is as following.
        kernel_size = base_dilated_rate + 1
        self.num_layers = 4
        self.conv = nn.ModuleList()
        new_dilated_rate = 1
        for k in range(self.num_layers):
            self.conv.append(nn.Conv2d(c_in, c_out, kernel_size=(1, kernel_size), dilation=(1, new_dilated_rate)))
            new_dilated_rate *= 2

    def forward(self, x):
        _out = 0.
        for i in range(self.num_layers):
            _out = self.conv[i](x)
            x = _out + x[..., :_out.shape[-1]]
        return _out


class gwnet(nn.Module):
    '''
    this model is for dynamic graph of every sample modeling.
    '''
    def __init__(self, device, num_nodes, dropout=0.3, supports=None, gcn_bool=True, addaptadj=True, aptinit=None,
                 in_dim=2, out_dim=12, residual_channels=32, dilation_channels=32, skip_channels=256, end_channels=512,
                 kernel_size=2, blocks=3, layers=2, dynamic_adj=False, seq_in_len=28):
        super(gwnet, self).__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj
        self.dynamic_adj = dynamic_adj

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()

        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1,1))
        self.supports = supports

        receptive_field = 1

        self.supports_len = 0
        if supports is not None:
            # if pre-defined graph
            self.supports_len += len(supports)

        if gcn_bool and addaptadj:
            # adaptive graph enable
            if aptinit is None:
                if supports is None:
                    self.supports = []
                self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10).to(device), requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).to(device), requires_grad=True).to(device)
                self.supports_len +=1
            else:
                if supports is None:
                    self.supports = []
                m, p, n = torch.svd(aptinit)
                initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
                initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
                self.nodevec1 = nn.Parameter(initemb1, requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(initemb2, requires_grad=True).to(device)
                self.supports_len += 1
        
        # ---------------------------- #
        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # dilated convolutions
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1,kernel_size),dilation=new_dilation))

                self.gate_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), dilation=new_dilation))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(residual_channels))
                new_dilation *=2
                receptive_field += additional_scope
                additional_scope *= 2
                if self.gcn_bool:
                    self.gconv.append(gcn(dilation_channels,residual_channels,dropout,support_len=self.supports_len))

        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                  out_channels=end_channels,
                                  kernel_size=(1,1),
                                  bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1,1),
                                    bias=True)
        c_out_dim = self.blocks * (kernel_size - 1) * (2 ** self.layers - 1)
        
        
        self.receptive_field = receptive_field
        
        if self.receptive_field > seq_in_len:
            self.linear_out = nn.Linear(receptive_field-c_out_dim, 1) # change the last dimension.
        else:
            self.linear_out = nn.Linear(seq_in_len-c_out_dim, 1) # change the last dimension.

    def forward(self, input, idx):
        
        self.supports = [] # for adjust for current framework

        in_len = input.size(3)
        if in_len<self.receptive_field:
            x = nn.functional.pad(input,(self.receptive_field-in_len,0,0,0))
        else:
            x = input
        x = self.start_conv(x)
        skip = 0

        # calculate the current adaptive adj matrix once per iteration
        new_supports = None
        if self.gcn_bool and self.addaptadj and self.supports is not None:
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            new_supports = self.supports + [adp]

        # WaveNet layers
        for i in range(self.blocks * self.layers):

            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + ------------->	*skip*

            #(dilation, init_dilation) = self.dilations[i]

            #residual = dilation_func(x, dilation, init_dilation, i)
            residual = x
            # dilated convolution
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate
            # parametrized skip connection
            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :,  -s.size(3):]
            except:
                skip = 0
            skip = s + skip
            # x: 32, 32, 137, 167
            if self.gcn_bool and self.supports is not None:
                if self.addaptadj:
                    x = self.gconv[i](x, new_supports)
                else:
                    x = self.gconv[i](x,self.supports)
            else:
                x = self.residual_convs[i](x)
            x = x + residual[:, :, :, -x.size(3):]
            x = self.bn[i](x)
            
        
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        x = self.linear_out(x)
        return x


class gwnet_oral(nn.Module):
    '''
    this model is for dynamic graph of every sample modeling.
    '''
    def __init__(self, device, num_nodes, dropout=0.3, supports=None, gcn_bool=True, addaptadj=True, aptinit=None,
                 in_dim=2, out_dim=12, residual_channels=32, dilation_channels=32, skip_channels=128, end_channels=256,
                 kernel_size=16, blocks=3, layers=2, dynamic_adj=False, time_len=168):
        super(gwnet_oral, self).__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj
        self.dynamic_adj = dynamic_adj

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()

        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1,1))
        self.supports = supports

        receptive_field = 1

        self.supports_len = 0
        if supports is not None:
            # if pre-defined graph
            self.supports_len += len(supports)

        if gcn_bool and addaptadj:
            # adaptive graph enable
            if aptinit is None:
                if supports is None:
                    self.supports = []
                self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10).to(device), requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).to(device), requires_grad=True).to(device)
                self.supports_len +=1
            else:
                if supports is None:
                    self.supports = []
                m, p, n = torch.svd(aptinit)
                initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
                initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
                self.nodevec1 = nn.Parameter(initemb1, requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(initemb2, requires_grad=True).to(device)
                self.supports_len += 1

        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # dilated convolutions
                # self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                #                                    out_channels=dilation_channels,
                #                                    kernel_size=(1,kernel_size),dilation=new_dilation))
                # self.gate_convs.append(nn.Conv2d(in_channels=residual_channels,
                #                                  out_channels=dilation_channels,
                #                                  kernel_size=(1, kernel_size), dilation=new_dilation))
                self.filter_convs.append(Big2SmallConv2d(c_in=residual_channels,
                                                         c_out=dilation_channels,
                                                         base_dilated_rate=new_dilation))

                self.gate_convs.append(Big2SmallConv2d(c_in=residual_channels,
                                                       c_out=dilation_channels,
                                                       base_dilated_rate=new_dilation))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(residual_channels))
                new_dilation *=2
                receptive_field += additional_scope
                additional_scope *= 2
                if self.gcn_bool:
                    self.gconv.append(gcn(dilation_channels,residual_channels,dropout,support_len=self.supports_len))

        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                  out_channels=end_channels,
                                  kernel_size=(1,1),
                                  bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1,1),
                                    bias=True)
        c_out_dim = self.blocks * (kernel_size - 1) * (2 ** self.layers - 1)
        import pdb; pdb.set_trace()
        self.linear_out = nn.Linear(time_len-c_out_dim, 1) # change the last dimension.
        self.receptive_field = receptive_field

    def forward(self, input):
        # if dynamic adj, the input need contains two inputs
        in_len = input.size(3)
        if in_len<self.receptive_field:
            x = nn.functional.pad(input,(self.receptive_field-in_len,0,0,0))
        else:
            x = input
        x = self.start_conv(x)
        skip = 0

        # calculate the current adaptive adj matrix once per iteration
        new_supports = None
        if self.gcn_bool and self.addaptadj and self.supports is not None:
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            new_supports = self.supports + [adp]

        # WaveNet layers
        for i in range(self.blocks * self.layers):

            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + ------------->	*skip*

            #(dilation, init_dilation) = self.dilations[i]

            #residual = dilation_func(x, dilation, init_dilation, i)
            residual = x
            # dilated convolution
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate
            # parametrized skip connection
            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :,  -s.size(3):]
            except:
                skip = 0
            skip = s + skip
            # x: 32, 32, 137, 167
            if self.gcn_bool and self.supports is not None:
                if self.addaptadj:
                    x = self.gconv[i](x, new_supports)
                else:
                    x = self.gconv[i](x,self.supports)
            else:
                x = self.residual_convs[i](x)
            x = x + residual[:, :, :, -x.size(3):]
            x = self.bn[i](x)

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        x = self.linear_out(x)
        return x