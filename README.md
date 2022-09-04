# Graph Influence

Graph Influence is an implementation of [Influence Functions](https://arxiv.org/abs/1703.04730) 
by Pang Wei Koh and Percy Liang for Graph Neural Networks. 
There are two main files in this repository: `train.py` and `graph_influence.py`. 
The first is a generic graph application to train graph neural networks on different toy datasets for different models. 
If you want to see more comprehensive information about influence functions,
please take a look at my [blog post about influence functions](https://rfdavid.com/influence-functions).

### Training from scratch

You can train a model from scratch leaving one specific node out for
validation/debugging purpose:  

`$ python train.py --dataset Cora --model GAT --leave_out 11`

| Parameter | Description | Default |
| ----------- | ----------- | ----------- |
| --dataset | Dataset to use (Cora, Pubmed, CiteSeer, Flickr) | Cora |
| --model | Network model (GCN, GAT, GIN, ARMA) | GCN | 
| --seed | Seeding number | 123 |
| --device | Device to train (cpu, cuda...) | cuda |
| --epochs | Number of epochs | 50 |
| --lr     | learning rate | 0.001 |
| --hidden\_layers | Number of hidden layers for MLP | 256 |
| --num\_layers | Number of layers (GCN or GIN only) | 2 |
| --heads | Attention heads | 8 |
| --leave\_out | Use this option to leave some specific node id out of the training | None |
| --node\_ids | Store the test loss of those specific node ids | None |
| --debug | Log level|  logging.INFO |


### Calculating the influence

To calculate the influence function use `graph_infliuence`. The following
example calculates the values for the test node ids 1708, 1720 and 1800 for
Cora dataset on GAT.

`$ python graph_influence.py --dataset Cora --model GAT --node_ids 1708 1720 1800`


| Parameter | Description | Default |
| ----------- | ----------- | ----------- |
| --dataset | Dataset to use (Cora, Pubmed, CiteSeer, Flickr) | Cora |
| --model | Network model (GCN, GAT, GIN, ARMA) | GCN | 
| --seed | Seeding number | 123 |
| --device | Device to train (cpu, cuda...) | cuda |
| --epochs | Number of epochs | 50 |
| --lr     | learning rate | 0.001 |
| --hidden\_layers | Number of hidden layers for MLP | 256 |
| --num\_layers | Number of layers (GCN or GIN only) | 2 |
| --heads | Attention heads | 8 |
| --node\_ids | Test node ids to measure the impact for each training node (required) | None |
| --recursion\_depth | Recursion depth for s\_test calculation | 1 |
| --r\_averaging | R averaging | 1 |
| --debug | Log level|  logging.INFO |


If you have any question feel free to contact me, open an issue or a pull
request.
