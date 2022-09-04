import random
import argparse
import torch
import logging
import os

from loader import load_data
from models.model import load_model
from influence import Influence
from utils import init_default_config, save_json 
import torch.nn.functional as F

def get_args() -> list:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Cora',
                        help='Dataset (Cora, Flickr, PubMed, CiteSeer)')
    parser.add_argument('--model', type=str, default='GCN',
                        help='Model (GCN, GAT, GIN, ARMA)')
    parser.add_argument('--hidden_layers', type=int, default=256,
                        help='Number of hidden layers')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of layers for GCN')
    parser.add_argument('--heads', type=int, default=8,
                        help='Number of heads for GAT')
    parser.add_argument('--seed', type=int, default=123, 
                        help='Random seed')
    parser.add_argument('--device', type=str, default='cuda', 
                        help='Device to train')
    parser.add_argument('--node_ids', nargs='+', type=int, required=True,
                        help='Testing node ids')
    parser.add_argument('--recursion_depth', type=int, default=1,
                        help='Recursion depth for s_test calculation')
    parser.add_argument('--r_averaging', type=int, default=1,
                        help='R averaging')
    parser.add_argument('--debug', dest="loglevel", action='store_const',
                        default=logging.INFO, const=logging.DEBUG, 
                        help='Display additional debug info')
    parser.add_argument('--experiment_name', type=str, default=False,
                        help='Experiment name to save the results')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = get_args()
    init_default_config(args)

    logging.info(f"Dataset: {args.dataset}")
    dataset = load_data(args.dataset)

    device = torch.device(args.device if torch.cuda.is_available() and
                            args.device != 'cpu'else 'cpu')
    logging.info(f"Using: {device}")
    
    model = load_model(args.model, 
            in_channels=dataset.num_features,
            hidden_channels=args.hidden_layers,
            heads=args.heads,
            num_layers=args.num_layers,
            out_channels=dataset.num_classes)
    model = model.to(device)
    logging.info(model)

    influence = Influence(model, dataset[0], device, args.recursion_depth, args.r_averaging)

    for node_id in args.node_ids:
        result = influence.calculate(node_id)
        if args.experiment_name:
            save_json(args.experiment_name, {
                'model': args.model,
                'dataset': args.dataset,
                'seed': args.seed,
                'node_id': node_id,
                'influence': result
                })
        print(result)
