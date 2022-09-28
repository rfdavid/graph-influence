import random
import argparse
import torch
import numpy as np
import logging
import os

from loader import load_data, sample_subgraph, display_subgraphs_info
from models.model import load_model
from utils import init_default_config, save_json
import torch.nn.functional as F

def get_args() -> list:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Cora',
                        help='Dataset (Cora, Flickr, PubMed, CiteSeer)')
    parser.add_argument('--model', type=str, default='GCN',
                        help='Model (GCN, GAT, GIN, ARMA)')
    parser.add_argument('--sampling', type=str, default=False,
                        help='Sampling method (shadowkhop, graphsaint, random)')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size for sampling')
    parser.add_argument('--seed', type=int, default=123, 
                        help='Random seed')
    parser.add_argument('--device', type=str, default='cuda', 
                        help='Device to train')
    parser.add_argument('--epochs', type=int, default=50, 
                        help='Number of epochs')
    parser.add_argument('--lr', type=int, default=0.001,
                        help='Learning rate')
    parser.add_argument('--hidden_layers', type=int, default=256,
                        help='Number of hidden layers')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of layers for GCN and GIN')
    parser.add_argument('--heads', type=int, default=8,
                        help='Number of heads for GAT')
    parser.add_argument('--leave_out', nargs='+', type=int, default=[],
                        help='Leave node x out')
    parser.add_argument('--node_ids', nargs='+', type=int, default=[],
                        help='Testing node ids to calculate the loss for comparison')
    parser.add_argument('--debug', dest="loglevel", action='store_const',
                        default=logging.INFO, const=logging.DEBUG, 
                        help='Display additional debug info')
    parser.add_argument('--experiment_name', type=str, default=False,
                        help='Experiment name to save the results')
    args = parser.parse_args()

    return args

def train_decoupled(model, batch, leave_out) -> float:
    total_loss = total_examples = 0
    for data in batch:
        data = data.to(device)
        if len(leave_out) > 0:
            for i in leave_out:
                data.train_mask[i] = False
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch, data.root_n_id)
        loss = F.cross_entropy(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * data.num_graphs
        total_examples += data.num_graphs

    return total_loss / total_examples


def train(model, data, leave_out) -> float:
    data = data.to(device)
    if len(leave_out) > 0:
        for i in leave_out:
            data.train_mask[i] = False
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

    return loss 


@torch.no_grad()
def test_decoupled(model, loader, device, node_ids, total_loss) -> list:
    model.eval()
    total_correct = total_examples = 0

    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch, data.root_n_id)
        total_correct += int((out.argmax(dim=-1) == data.y).sum())
        total_examples += data.num_graphs
    
    accs = [0, 0, total_correct / total_examples]

    return accs, 0


@torch.no_grad()
def test(model, data, device, node_ids, total_loss) -> list:
    model.eval()
    total_correct = total_examples = 0

    out = model(data.x.to(device), data.edge_index.to(device))
    pred = out.argmax(dim=-1)
    correct = pred.eq(data.y.to(device))

    # loss at testing point x
    for node_id in node_ids:
        prediction = out[node_id].view(1, -1)
        label = torch.tensor([data.y[node_id]]).to(device)
        loss = F.cross_entropy(prediction, label)
        total_loss[node_id] += float(loss)

    accs = []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        accs.append(correct[mask].sum().item() / mask.sum().item())

    return accs,total_loss


def run_train(model, args, train_data, test_data=None) -> None:
    max_test_acc = 0
    total_loss = {k: 0 for k in args.node_ids}

    for epoch in range(0, args.epochs):
        if test_data:
            loss = train_decoupled(model, train_data, args.leave_out)
            accs,total_loss = test_decoupled(model, test_data, args.device, args.node_ids, total_loss)
        else:
            loss = train(model, train_data, args.leave_out)
            accs,total_loss = test(model, train_data, args.device, args.node_ids, total_loss)

        max_test_acc = max(max_test_acc, accs[2])
        logging.info(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Train: {accs[0]:.4f}, '
                f'Val: {accs[1]:.4f}, Test: {accs[2]:.4f}, Max Test: {max_test_acc:.4f}')
    logging.info(f'Total loss at testing points: {total_loss}')

    if args.experiment_name:
        save_json(args.experiment_name, {
            'model': args.model,
            'dataset': args.dataset,
            'seed': args.seed,
            'leave_out': args.leave_out, 
            'epochs': args.epochs,
            'max_testing_accuracy': max_test_acc,
            'total_loss_for_testing_nodes': total_loss
            })


if __name__ == '__main__':
    args = get_args()
    init_default_config(args)

    logging.info(f'Dataset: {args.dataset}')
    dataset = load_data(args.dataset)

    train_sg = []
    test_sg = []

    if args.sampling:
        train_l, test_l, val_l = sample_subgraph(dataset, args.sampling, args.batch_size)
        # For reproducibility
        for d in train_l:
            if args.sampling == 'shadowkhop':
                d.train_mask[d.train_mask==False] = True
            train_sg.append(d)

        for d in test_l:
            if args.sampling == 'shadowkhop':
                d.test_mask[d.test_mask==False] = True
            test_sg.append(d)


    device = torch.device(args.device if torch.cuda.is_available() and
                            args.device != 'cpu'else 'cpu')
    logging.info(f'Using: {device}')

    logging.info("Training Sub-graphs:")
    for s in train_sg:
        logging.info(s)
    
    model = load_model(args.model, 
            in_channels=dataset.num_features,
            hidden_channels=args.hidden_layers,
            num_layers=args.num_layers,
            heads=args.heads,
            out_channels=dataset.num_classes)
    model = model.to(device)
    logging.info(f'Model: {model}')
    logging.info(f'Testing node ids to calculate the loss: {args.node_ids}')

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    if args.sampling:
        run_train(model, args, train_sg, test_sg)
    else:
        run_train(model, args, dataset[0])
