---
title: Visualization of pruned networks
categories: [python, machine_learning]
tag: [Visualization,Pruning,networkx]
author: [<kanishk>, <lipika>, <ksheer>]
---

## Visualization of structured pruning
We have used the python library ‘networkx’ to visualize the process of structured pruning on a neural network. The aim of this activity is to verify the accuracy of the structured pruning algorithm defined above. So, we begin by creating a fully connected neural network and assigning weights to each of the input edges in each layer manually. Refer to the code snippet below for the construction of the neural network.


```python
import torch
import torch.nn as nn
import pruning_methods as pm


# Create a custom neural network using nn.Sequential
model = nn.Sequential(
    nn.Linear(2, 2),
    nn.ReLU(),
    nn.Linear(2, 4),
    nn.ReLU(),
    nn.Linear(4, 2),
    nn.ReLU()
)


# Manually set custom weight matrices and biases
model[0].weight.data = torch.tensor([[1, -1.0], [5, 2.0]])
model[0].bias.data = torch.tensor([0.1, 0.2])


model[2].weight.data = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]])
model[2].bias.data = torch.tensor([-0.2, 0.1, 0.3, 0.5])


model[4].weight.data = torch.tensor([[0.1, -0.2, 0.3, 0.1], [-0.1, 0.8, 0.1, -0.4]])
model[4].bias.data = torch.tensor([0.1, -0.2])


# Deep copy the model
model_cpy = copy.deepcopy(model)
plot_network(model)
```

After creating this neural network, we have used the `networkx` library to visualize the neural network. The resultant visualization will have a node corresponding to each node in the network. The nodes of one layer are aligned vertically. The nodes of different layers are connected with edges. We have displayed the edge weight of each edge in the plot along with color coding the edges, i.e., the color of an edge denoted its weight with respect to other edges. 

![Pruning](/assets/img/pruning-visualization-1.png)

After this, we use one_shot structured pruning to prune/remove 50% of the nodes in each layer except input and output layers. We use the pruning algorithm defined above. Ideally, after pruning following should happen:
* One out of two neurons in layer 1(0 indexed) should be removed. The L1-norm of the parameters of the removed node should be less than that of the unpruned node.

```
L1 Norm of Node 0 = |bias| + |edge_weight_0| + |edge_weight_1| 
                  => 0.1 + 1 + 1 = 2.1

L1 Norm of Node 1 = |bias| + |edge_weight_0| + |edge_weight_1| 
                  => 0.2 + 5 + 2 = 7.2
```

As it can be seen from above two equations, Node 0 should be removed from layer 0. 

* Two out of four neurons should be removed from layer 2 with least L1 norm of their parameters.

```
L1 Norm of Node 0 = |bias| + |edge_weight_0| + |edge_weight_1| 
                  => 0.2 + 0.1 + 0.2 = 0.5
L1 Norm of Node 1 = |bias| + |edge_weight_0| + |edge_weight_1| 
                  => 0.1 + 0.3 + 0.4 = 0.8
L1 Norm of Node 2 = |bias| + |edge_weight_0| + |edge_weight_1| 
                  => 0.3 + 0.5 + 0.6 = 1.4
L1 Norm of Node 3 = |bias| + |edge_weight_0| + |edge_weight_1| 
                  => 0.5 + 0.7 + 0.8 = 2.0
```
As it can be seen from above equations, Node 0 and Node 1 should be removed. 

We can observe in the figure below, that the desired state of the neural network has been achieved after pruning. 

![Pruning](/assets/img/pruning-visualization-2.png)