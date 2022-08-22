import random
import argparse
import torch
import logging
import os

from loader import load_data
from model import load_model
from influence import Influence
from utils import init_default_config, save_json 
import torch.nn.functional as F

def get_args() -> list:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Cora',
                        help='Dataset (Cora, Flickr, Reddit)')
    parser.add_argument('--model', type=str, default='GCN',
                        help='Model (GCN, GAT)')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of layers for GCN')
    parser.add_argument('--seed', type=int, default=123, 
                        help='Random seed')
    parser.add_argument('--device', type=str, default='cuda', 
                        help='Device to train')
    parser.add_argument('--node_id', type=int, default=50, 
                        help='Testing node id')
    parser.add_argument('--recursion_depth', type=int, default=1,
                        help='Recursion depth for s_test calculation')
    parser.add_argument('--r_averaging', type=int, default=1,
                        help='R averaging')
    parser.add_argument('--debug', dest="loglevel", action='store_const',
                        default=logging.INFO, const=logging.DEBUG, 
                        help='Display additional debug info')
    parser.add_argument('--experiment_name', type=str, default='default',
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
            hidden_channels=256,
            num_layers=args.num_layers,
            out_channels=dataset.num_classes)
    model = model.to(device)
    logging.info(model)

    influence = Influence(model, dataset[0], device, args.recursion_depth, args.r_averaging)
    result = influence.calculate(args.node_id)
    save_json(args.experiment_name, result)
    print(result)
