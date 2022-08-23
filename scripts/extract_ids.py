"""
    Extract training and testing node ids from a graph dataset
"""

import os.path as osp
import torch
import os
import sys
import argparse

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from loader import load_data

def get_args() -> list:
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', default=False,
                       help='Display all test node ids')
    parser.add_argument('--train', action='store_true', default=False,
                       help='Display all train node ids')
    parser.add_argument('--dataset', type=str, default='Cora',
                        help='Dataset (Cora, Flickr, Reddit...)')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = get_args()

    dataset = load_data(args.dataset)
    data = dataset[0]  # Get the first graph object.

    sys.stderr.write(f'Dataset: {dataset}:\n')
    sys.stderr.write('==================\n')
    sys.stderr.write(f'Number of graphs: {len(dataset)}\n')
    sys.stderr.write(f'Number of features: {dataset.num_features}\n')
    sys.stderr.write(f'Number of classes: {dataset.num_classes}\n')

    # Gather some statistics about the graph.
    sys.stderr.write(f'Number of nodes: {data.num_nodes}\n')
    sys.stderr.write(f'Number of edges: {data.num_edges}\n')
    sys.stderr.write(f'Average node degree: {data.num_edges / data.num_nodes:.2f}\n')
    sys.stderr.write(f'Number of training nodes: {data.train_mask.sum()}\n')
    sys.stderr.write(f'Number of testing nodes: {data.test_mask.sum()}\n')
    sys.stderr.write(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.3f}\n')
    sys.stderr.write(f'Has isolated nodes: {data.has_isolated_nodes()}\n')
    sys.stderr.write(f'Has self-loops: {data.has_self_loops()}\n')
    sys.stderr.write(f'Is undirected: {data.is_undirected()}\n')

    if args.train:
        train_ids = torch.where(data.train_mask == True)
        for id in train_ids[0]:
            print(int(id))

    if args.test:
        test_ids = torch.where(data.test_mask == True)
        for id in test_ids[0]:
            print(int(id))
