from torch_geometric.loader import ShaDowKHopSampler, GraphSAINTRandomWalkSampler, RandomNodeSampler
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import Flickr
from torch_geometric.transforms import NormalizeFeatures
import os.path as osp

def load_data(dataset):
    if dataset == 'Flickr':
        path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Flickr')
        dataset = Flickr(path)
    elif dataset == 'Cora':
        dataset = load_planetoid('Cora')
    elif dataset == 'PubMed':
        dataset = load_planetoid('PubMed')
    elif dataset == 'CiteSeer':
        dataset = load_planetoid('CiteSeer')
    else:
        raise Exception(f'Invalid dataset {dataset}')

    return dataset

def load_planetoid(dataset_name):
    return Planetoid(root='data/Planetoid', name=dataset_name, transform=NormalizeFeatures())

def sample_subgraph(dataset, sampling_method, batch_size):
    sampling_method = sampling_method.lower()
    train_loader = val_loader = test_loader = None
    data = dataset[0]

    if sampling_method == 'graphsaint':
        train_loader = GraphSAINTRandomWalkSampler(data, batch_size=batch_size, walk_length=2,
                num_steps=15, sample_coverage=100,
                save_dir=dataset.processed_dir,
                num_workers=1)

    if sampling_method == 'random':
        train_loader = RandomNodeSampler(data, num_parts=100, shuffle=True, num_workers=1)

    if sampling_method == 'shadowkhop':
        kwargs = {'batch_size': batch_size, 'num_workers': 1, 'persistent_workers': True}
        train_loader = ShaDowKHopSampler(data, depth=2, num_neighbors=5,
                node_idx=data.train_mask, **kwargs)
        val_loader = ShaDowKHopSampler(data, depth=2, num_neighbors=5,
                node_idx=data.val_mask, **kwargs)
        test_loader = ShaDowKHopSampler(data, depth=2, num_neighbors=5,
                node_idx=data.test_mask, **kwargs)

        # for reproducibility
        train_sg = []
        test_sg = []
        val_sg = []
        for d in train_loader:
#            d.train_mask[d.train_mask==False] = True
            train_sg.append(d)

        for d in test_loader:
#            d.test_mask[d.test_mask==False] = True
            test_sg.append(d)

        for d in val_loader:
#            d.test_mask[d.test_mask==False] = True
            test_sg.append(d)

        train_loader = train_sg
        test_loader = test_sg
        val_loader = val_sg

    return train_loader, test_loader, val_loader

def display_subgraphs_info(loader):
    for i,d in enumerate(loader):
        print(f'{i}: {d}')

    print("Total subgraphs batch:", len(loader))
