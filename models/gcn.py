import torch
from torch.nn import Linear
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GraphConv
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool

class GCN(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, out_channels, num_layers):
        super(GCN, self).__init__()
        self.num_layers = num_layers

        if num_layers > 1:
            self.first_conv = GCNConv(input_channels, hidden_channels)

        if num_layers > 2:
            self.convs = torch.nn.ModuleList()
            for layer in range(num_layers - 2):
                self.convs.append(GCNConv(hidden_channels, hidden_channels))

        if num_layers == 1:
            hidden_channels = input_channels

        self.last_conv = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        if self.num_layers > 1:
            x = self.first_conv(x, edge_index)
            x = x.relu()

        if self.num_layers > 2:
            for conv in self.convs:
                x = conv(x, edge_index)
                x = x.relu()

        x = self.last_conv(x, edge_index)

        return x.log_softmax(dim=-1)
