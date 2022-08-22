import random
import argparse
import torch
import numpy as np
import logging
import os

from torch_geometric import seed_everything
from loader import load_data, sample_subgraph, display_subgraphs_info
from model import load_model
import torch.nn.functional as F

def get_args() -> list:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Cora',
                        help='Dataset (Cora, Flickr, Reddit)')
    parser.add_argument('--model', type=str, default='GCN',
                        help='Model (GCN, GAT)')
    parser.add_argument('--seed', type=int, default=123, 
                        help='Random seed')
    parser.add_argument('--device', type=str, default='cuda', 
                        help='Device to train')
    parser.add_argument('--epochs', type=int, default=50, 
                        help='Number of epochs')
    parser.add_argument('--lr', type=int, default=0.001,
                        help='Learning rate')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of layers')
    parser.add_argument('--leave_out', type=int, default=False,
                        help='Leave node x out')
    parser.add_argument('--debug', dest="loglevel", action='store_const',
                        default=logging.INFO, const=logging.DEBUG, 
                        help='Display additional debug info')
    args = parser.parse_args()

    return args


def train(model, data, leave_out) -> float:
    if leave_out:
        data.train_mask[leave_out] = False
    model.train()
    data = data.to(device)
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

    return loss 


@torch.no_grad()
def test(model, data) -> list:
    model.eval()
    total_correct = total_examples = 0

    out = model(data.x.to(device), data.edge_index.to(device))
    pred = out.argmax(dim=-1)
    correct = pred.eq(data.y.to(device))

    accs = []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        accs.append(correct[mask].sum().item() / mask.sum().item())

    return accs


def run_train(model, data, epochs, leave_out) -> None:
    max_test_acc = 0

    for epoch in range(0, epochs):
        loss = train(model, data, leave_out)
        accs = test(model, data)
        max_test_acc = max(max_test_acc, accs[2])
        logging.info(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Train: {accs[0]:.4f}, '
                f'Val: {accs[1]:.4f}, Test: {accs[2]:.4f}, Max Test: {max_test_acc:.4f}')

if __name__ == '__main__':
    args = get_args()

    seed_everything(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    log_format = '%(asctime)s: %(message)s'
    logging.basicConfig(level=args.loglevel, format=log_format)

    logging.info(f"Dataset: {args.dataset}")
    dataset = load_data(args.dataset)

    device = torch.device(args.device if torch.cuda.is_available() and
                            args.device != 'cpu'else 'cpu')
    logging.info(f"Using: {device}")
    
    model = load_model(args.model, 
            in_channels=dataset.num_features,
            hidden_channels=256,
            num_layers=args.num_layers,
            out_channels=dataset.num_classes)
    model = model.to(device)
    logging.info(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    run_train(model, dataset[0], args.epochs, args.leave_out)
