from torch_geometric.loader import ShaDowKHopSampler, GraphSAINTRandomWalkSampler
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
    if sampling_method == 'GraphSAINT':
        loader = GraphSAINTRandomWalkSampler(dataset[0], batch_size=batch_size, walk_length=2,
                                            num_steps=15, sample_coverage=100,
                                            save_dir=dataset.processed_dir,
                                            num_workers=1)
    return loader

def display_subgraphs_info(loader):
    for i,d in enumerate(loader):
        print(f'{i}: {d}')

    print("Total subgraphs batch:", len(loader))
