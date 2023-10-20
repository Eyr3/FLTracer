import torch
import torch.nn as nn
class KalmanFilter_Estimator(nn.Module):
    def __init__(self,layer_num=14,dim_num=1)-> object:
        super(KalmanFilter_Estimator, self).__init__()
        self.layer_num=layer_num
        self.dim_num = dim_num
        if dim_num == 1:
            self.u = nn.Parameter(torch.randn(1,layer_num))
            self.k1 = nn.Linear(layer_num, layer_num, bias=False)
            self.k2 = nn.Linear(layer_num, layer_num, bias=False)
            self.k5 = nn.Linear(layer_num, layer_num, bias=False)
            self.w = nn.Parameter(torch.randn(layer_num))
            self.k7 = nn.Linear(layer_num, layer_num, bias=False)
            self.h1_and_v1 = nn.Linear(layer_num, layer_num)
            self.h4_and_v2 = nn.Linear(layer_num, layer_num)

            self.k3 = nn.Parameter(torch.zeros(layer_num,layer_num),requires_grad=False)
            self.k4 = nn.Parameter(torch.ones(layer_num,layer_num), requires_grad=False)
            self.k6 = nn.Parameter(torch.ones(layer_num, layer_num), requires_grad=False)
            self.k8 = nn.Parameter(torch.zeros(layer_num), requires_grad=False)
            self.h3 = nn.Parameter(torch.zeros(layer_num,layer_num),requires_grad=False)
            self.h4 = nn.Parameter(torch.zeros(layer_num,layer_num),requires_grad=False)
        else:
            self.u = nn.Parameter(torch.randn(layer_num, dim_num))
            self.k1 = nn.Parameter(torch.randn(layer_num, dim_num))
            self.k2 = nn.Parameter(torch.randn(layer_num, dim_num))
            self.k5 = nn.Parameter(torch.randn(layer_num, dim_num))
            self.w = nn.Parameter(torch.randn(dim_num))
            self.k7 = nn.Parameter(torch.randn(layer_num, dim_num))
            self.h1 = nn.Parameter(torch.randn(layer_num, dim_num))
            self.h4 = nn.Parameter(torch.randn(layer_num, dim_num))
            self.v1 = nn.Parameter(torch.randn(layer_num, dim_num))
            self.v2 = nn.Parameter(torch.randn(layer_num, dim_num))

            self.k3 = nn.Parameter(torch.zeros(layer_num, dim_num), requires_grad=False)
            self.k4 = nn.Parameter(torch.ones(layer_num, dim_num), requires_grad=False)
            self.k6 = nn.Parameter(torch.ones(layer_num, dim_num), requires_grad=False)
            self.k8 = nn.Parameter(torch.zeros(layer_num), requires_grad=False)
            self.h2 = nn.Parameter(torch.zeros(layer_num, dim_num), requires_grad=False)
            self.h3 = nn.Parameter(torch.zeros(layer_num, dim_num), requires_grad=False)

    def forward(self, x_t_1 , u_t_0):
        #x:position-local update,velocity-global update
        if self.dim_num == 1:
            g_t_1 = x_t_1[:, int(x_t_1.shape[1] / 2):]
            u_t_1 = x_t_1[:, :int(x_t_1.shape[1] / 2)]
            g_t = g_t_1 + self.u + self.k7(u_t_0)
            u_t = self.k1(u_t_1)+self.k2(g_t_1)+self.k5(self.u)+self.w
            u_z_t = self.h1_and_v1(u_t)
            g_z_t = self.h4_and_v2(g_t)
            z_t = torch.cat((u_z_t,g_z_t),dim=1)
            x_t = torch.cat((u_t,g_t),dim=1)
        else:
            g_t_1 = x_t_1[:, int(x_t_1.shape[1] / 2):]
            u_t_1 = x_t_1[:, :int(x_t_1.shape[1] / 2)]
            g_t = g_t_1 + self.u + self.k7*u_t_0
            u_t = self.k1*u_t_1 + self.k2*g_t_1 + self.k5*self.u + self.w
            u_z_t = self.h1*u_t + self.v1
            g_z_t = self.h4*g_t + self.v2
            z_t = torch.cat((u_z_t, g_z_t), dim=1)
            x_t = torch.cat((u_t, g_t), dim=1)
        return x_t,z_t
    def update(self, x_t_1 , u_t_0):
        if self.dim_num == 1:
            g_t_1 = x_t_1[:, int(x_t_1.shape[1] / 2):]
            u_t_1 = x_t_1[:, :int(x_t_1.shape[1] / 2)]
            g_t = g_t_1 + self.u + self.k7(u_t_0)
            u_t = self.k1(u_t_1) + self.k2(g_t_1) + self.k5(self.u) + self.w
            x_t = torch.cat((u_t, g_t), dim=1)
        else:
            g_t_1 = x_t_1[:, int(x_t_1.shape[1] / 2):]
            u_t_1 = x_t_1[:, :int(x_t_1.shape[1] / 2)]
            g_t = g_t_1 + self.u + self.k7 * u_t_0
            u_t = self.k1 * u_t_1 + self.k2 * g_t_1 + self.k5 * self.u + self.w
            x_t = torch.cat((u_t, g_t), dim=1)
        return x_t