import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class FourierEnc(nn.Module):
    def __init__(self, input_dim, enc_dim):
        super().__init__()
        self.input_dim = input_dim
        self.enc_dim = enc_dim

        # use Gaussian
        self.B = torch.randn((self.input_dim, self.enc_dim)) * 2 * np.pi

    def forword(self,x):
        fm = torch.cat((torch.cos(x @ self.B), torch.sin(x @ self.B)), dim=-1)
        return fm


class FFN(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, skip_in=()):
        super().__init__()

        dims = [dim_in] + dim_hidden + [dim_out]
        self.num_layers = len(dims)
        self.skip_in = skip_in

        self.layers = nn.ModuleList([])
        self.layers.append(nn.Linear(dims[1]*2,dims[2]))

        for layer in range(2, self.num_layers - 1):

            if layer + 1 in skip_in:
                out_dim = dims[layer + 1] - dim_in
            else:
                out_dim = dims[layer + 1]

            self.layers.append(nn.Linear(dims[layer], out_dim))

        self.activation = nn.Softplus(beta=100)

        #Fourier Gaussian
        self.B = torch.randn((dim_in, dims[1])) * 2 * np.pi

        print(self)

    def fouriermap(self,x):
        return torch.cat((torch.cos(x @ self.B), torch.sin(x @ self.B)), dim=-1)


    def forward(self, input):

        x = self.fouriermap(input)

        layer_index = 1
        for layer in self.layers:

            if layer_index in self.skip_in:
                x = torch.cat([x, input], -1) / np.sqrt(2)

            x = layer(x)

            if layer_index < self.num_layers - 2:
                x = self.activation(x)

            layer_index = layer_index + 1

        return x


# # test
# mynet = FFN(3,[256,256,256,256],1,[3])
# a = mynet(torch.tensor([1,2,3]).float())
# print(a)
