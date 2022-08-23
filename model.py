from gcn import GCN
from gat import GAT

def load_model(model, **kwargs):
    if model == 'GCN':
        model = GCN(kwargs['in_channels'], kwargs['hidden_channels'],
                kwargs['out_channels'], kwargs['num_layers'])
    elif model == 'GAT':
        model = GAT(kwargs['in_channels'], kwargs['hidden_channels'],
                kwargs['out_channels'], kwargs['heads'])
    else:
        raise Exception(f'Invalid model {model}')

    return model
