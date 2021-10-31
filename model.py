import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys

# model
class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncvl,vw->ncwl',(x,A))
        return x.contiguous()

class Aconv(nn.Module):
    def __init__(self):
        super(Aconv,self).__init__()

    def forward(self,x, A, shift):
        Align_x = torch.roll(x, shift, dims=3) 
        out = torch.zeros_like(x).to(x.device)
        Align_x[...,:shift] = out[...,:shift]  
        x = torch.einsum('ncvl,vw->ncwl',(Align_x,A))
        return x.contiguous()

class linear(nn.Module):
    def __init__(self,c_in,c_out):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)

    def forward(self,x):
        return self.mlp(x)

class emb_trans(nn.Module):
    def __init__(self, device, n_dim=10):
        super(emb_trans, self).__init__()
        self.w = nn.Parameter(torch.eye(n_dim).to(device), requires_grad=True).to(device)
        self.b = nn.Parameter(torch.zeros(n_dim).to(device), requires_grad=True).to(device)
    def forward(self, nodevec1, nodevec2, n):
        nodevec1 = nodevec1.mm(self.w) + self.b.repeat(n, 1)
        nodevec2 = (nodevec2.T.mm(self.w) + self.b.repeat(n, 1)).T
        return nodevec1, nodevec2

class Agcn(nn.Module):
    def __init__(self,c_in,c_out,dropout, kernel_size, dilation, device, e_dim=10):
        super(Agcn,self).__init__()
        self.Aconv = Aconv()
        c_in = (1+kernel_size)*c_in
        self.mlp = linear(c_in,c_out)
        self.dropout = dropout
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.e_trans = nn.ModuleList()

        for i in range(kernel_size-1):
            self.e_trans.append(emb_trans(device, e_dim))

    def forward(self,x, nodevec1, nodevec2):
        out = [x]
        shift = 0
        n = nodevec1.size(0)
        
        adp = F.softmax(F.relu(torch.mm(nodevec1, nodevec2)), dim=1)
        x1 = self.Aconv(x, adp, shift)
        out.append(x1) 
        shift = self.dilation
        x2 = x1  
        for i in range(self.kernel_size-1):
            nodevec1, nodevec2 = self.e_trans[i](nodevec1, nodevec2, n)
            adp = F.softmax(F.relu(torch.mm(nodevec1, nodevec2)), dim=1)
            x1 = self.Aconv(x2, adp, shift)
            out.append(x1)
            shift = shift + self.dilation
            x2 = x1  
        h = torch.cat(out,dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h

class Anet(nn.Module):
    def __init__(self, device, num_nodes, dropout=0.3, aptinit=None, in_dim=2,out_dim=12,
                 residual_channels=32,dilation_channels=32,skip_channels=256,end_channels=512,kernel_size=2,blocks=4,layers=2, e_dim=10, kernel_size_Agcn=2):
        super(Anet, self).__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers

        self.align = nn.ModuleList()
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()
        self.e_trans = nn.ModuleList()

        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1,1))
    
        receptive_field = 1
        if aptinit is None:
            self.nodevec1 = nn.Parameter(torch.randn(num_nodes, e_dim).to(device), requires_grad=True).to(device)
            self.nodevec2 = nn.Parameter(torch.randn(e_dim, num_nodes).to(device), requires_grad=True).to(device)
        else:
            m, p, n = torch.svd(aptinit)
            initemb1 = torch.mm(m[:, :e_dim], torch.diag(p[:e_dim] ** 0.5))
            initemb2 = torch.mm(torch.diag(p[:e_dim] ** 0.5), n[:, :e_dim].t())
            self.nodevec1 = nn.Parameter(initemb1, requires_grad=True).to(device)
            self.nodevec2 = nn.Parameter(initemb2, requires_grad=True).to(device)

        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # dilated convolutions
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1,kernel_size),dilation=new_dilation))

                self.gate_convs.append(nn.Conv1d(in_channels=residual_channels,
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
                self.gconv.append(Agcn(residual_channels, dilation_channels, dropout, kernel_size_Agcn, new_dilation,device, e_dim = e_dim))
                self.e_trans.append(emb_trans(device, e_dim))
                
                new_dilation *=2
                receptive_field += additional_scope
                additional_scope *= 2
                

        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                  out_channels=end_channels,
                                  kernel_size=(1,1),
                                  bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1,1),
                                    bias=True)

        self.receptive_field = receptive_field



    def forward(self, input):
        in_len = input.size(3)
        if in_len<self.receptive_field:
            x = nn.functional.pad(input,(self.receptive_field-in_len,0,0,0))
        else:
            x = input
        x = self.start_conv(x)
        skip = 0


        n1 = self.nodevec1
        n2 = self.nodevec2
        n = self.nodevec1.size(0)
        for i in range(self.blocks * self.layers):
            residual = x   # residual_channels
            filter = self.filter_convs[i](residual)  # dilation_channels
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)  # dilation_channels
            gate = torch.sigmoid(gate)
            x = filter * gate           

            ss = x
            x2 = self.gconv[i](residual, n1, n2)  # residual_channels -> dilation_channels
            n1, n2 = self.e_trans[i](n1, n2, n)
            ss = ss + x2[:, :, :, -x.size(3):]  
            # parametrized skip connection
            
            s = self.skip_convs[i](ss)  # skip_channels
            try:
                skip = skip[:, :, :,  -s.size(3):]
            except:
                skip = 0
            skip = s + skip

            x = self.residual_convs[i](ss)
            x = x + residual[:, :, :, -x.size(3):]
            x = self.bn[i](x)

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x