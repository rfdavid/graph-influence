# Graph Influence

Graph Influence is an implementation of [Influence Functions](https://arxiv.org/abs/1703.04730) by Pang Wei Koh and Percy Liang for Graph Neural Networks.

## Getting started

### Training from scratch

You can train a model from scratch leaving one specific node out for
validation/debugging purpose:  

`$ python train.py --dataset Cora --model GAT --leave_out 11`

### Calculating the influence

To calculate the influence function use the following command. 
