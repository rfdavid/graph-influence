import torch
import torch.nn.functional as F
from torch_geometric.nn import ARMAConv

class ARMA(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        
        self.conv1 = ARMAConv(in_channels, hidden_channels, num_stacks=3,
                              num_layers=2, shared_weights=True, dropout=0.25)

        self.conv2 = ARMAConv(hidden_channels, out_channels, num_stacks=3,
                              num_layers=2, shared_weights=True, dropout=0.25,
                              act=lambda x: x)
        
    def forward(self, x, edge_index):
        x = F.dropout(x, training=self.training)
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
