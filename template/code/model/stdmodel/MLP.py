import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class MLPNet(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, skip_in=(), final_activation = None):
        super().__init__()

        dims = [dim_in] + dim_hidden + [dim_out]
        self.num_layers = len(dims)
        self.skip_in = skip_in

        self.layers = nn.ModuleList([])

        for layer in range(0, self.num_layers - 1):

            if layer + 1 in skip_in:
                out_dim = dims[layer + 1] - dim_in
            else:
                out_dim = dims[layer + 1]

            self.layers.append(nn.Linear(dims[layer], out_dim))

            # # if true preform preform geometric initialization
            # if geometric_init:

            #     if layer == self.num_layers - 2:

            #         torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[layer]), std=0.00001)
            #         torch.nn.init.constant_(lin.bias, -radius_init)
            #     else:
            #         torch.nn.init.constant_(lin.bias, 0.0)
            #         torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

        if True:
            self.activation = nn.Softplus(beta=100)

        # vanilla relu
        else:
            self.activation = nn.ReLU()
        
        print(self)

    def forward(self, input):

        x = input

        layer_index = 0
        for layer in self.layers:

            if layer_index in self.skip_in:
                x = torch.cat([x, input], -1) / np.sqrt(2)

            x = layer(x)

            if layer_index < self.num_layers - 2:
                x = self.activation(x)
            
            layer_index+=1

        return x

# test 
mynet = MLPNet(3,[256,256,256,256,256],1,[4])
mynet(torch.tensor([12,1,2]).float())
print('done')