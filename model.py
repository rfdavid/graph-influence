from gcn import GCN

def load_model(model, **kwargs):
    if model == 'GCN':
        model = GCN(kwargs['in_channels'], kwargs['hidden_channels'],
                kwargs['out_channels'], kwargs['num_layers'])
    else:
        raise Exception(f'Invalid model {model}')

    return model
