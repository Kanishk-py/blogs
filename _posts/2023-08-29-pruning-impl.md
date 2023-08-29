---
title: Implementing structured pruning in python
categories: [python, machine_learning]
tag: [Pruning,implementation]
author: [<kanishk>, <lipika>, <ksheer>]
---

# Pruning
Pruning is a technique in machine learning that reduces the size of the model by removing the unnecessary parameters. It is a technique to reduce the size of the model and the computation required to train the model. Pruning can be done in two ways:
* Structured
* Unstructured

I have discussed about the pruning techniques in detail in the following blog post [Pruning Techniques]({% link _posts/2023-08-29-pruning-types.md %})

## Implementation

For our implementation let's say we take a overly complicated neural network with 5 layers. The size of each layer is as follows:

```python
model = nn.Sequential(
    nn.Linear(2, 20),
    nn.ReLU(),
    nn.Linear(20, 18),
    nn.ReLU(),
    nn.Linear(18, 16),
    nn.ReLU(),
    nn.Linear(16, 14),
    nn.ReLU(),
    nn.Linear(14, 2),
    nn.Sigmoid()
)
```

For implementing one shot pruning, we can define some utility functions that will help us in the process. 

```python
# Training loop
def train(model,X_train_tensor, X_val_tensor,y_train_tensor, y_val_tensor, epochs = 100):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor)
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")

def get_data(weight_data, layer_shape, layers_pruned):
    return weight_data[:, [col for col in range(layer_shape[1]) if col not in layers_pruned]]

def add_layer(unpruned_layers, input_shape, output_shape, layer_data, activation = nn.ReLU()):
	layer = nn.Linear(input_shape, output_shape)
	with torch.no_grad():
		layer.data = layer_data
	unpruned_layers.append(layer)
	unpruned_layers.append(activation)
```

`get_data()`
: This function takes in the weight data, layer shape and the layers pruned as input and returns the weight data of the unpruned layers.

`add_layer()`
: This function takes in the unpruned layers, input shape, output shape, layer data and activation function as input and adds the layer to the unpruned layers, with edges having their assigned weights.

### One-shot Pruning (Reinitialization with Trained Weights)

```python
# In this method, we prune prune_ratio fatures in each layer
# The nn.Sequential method randomly initialises when called
def oneshot_pruning( post_training_model, input_shape, output_shape, prune_ratio = 0.2):
    unpruned_layers = [] 
    layer_index = 0
    layers_pruned = []
    for name, param in post_training_model.named_parameters():
        if 'weight' in name:
            # Not pruning the last output layer
            if layer_index == len(post_training_model)-2:
                add_layer(unpruned_layers,
			input_shape,
			output_shape,
			get_data(post_training_model[layer_index].weight.data,
				param.data.shape,
				layers_pruned),
			nn.Sigmoid())
                continue
            # Sorting the features in a layer based on l1 norm
            param_with_skipped_input = get_data(post_training_model[layer_index].weight.data, param.data.shape, layers_pruned)
            sorted_layers = torch.linalg.norm(param_with_skipped_input, ord=1, dim=1).argsort(dim=-1)
            layers_not_pruned = sorted(sorted_layers[int(prune_ratio*param_with_skipped_input.shape[0]):])
            layers_pruned = sorted(sorted_layers[:int(prune_ratio*param_with_skipped_input.shape[0])])

            # Initialising unpruned neurons with pre-training values
            layer_data = param_with_skipped_input[layers_not_pruned, :] 
            add_layer(unpruned_layers, input_shape, layer_data.shape[0], layer_data)
            input_shape = layer_data.shape[0]
            #skipping every alternate relu layer
            layer_index=layer_index+2
    return nn.Sequential(*unpruned_layers)  
```

`if statement at line 10`
: We don't prune the last layer because we have a classification model. `add_layer()` function is called to add the last layer to the unpruned layers.

`if not the last layer (after line 20)`
: We sort the features in a layer based on l1 norm. We then remove the features with the lowest l1 norm. We store the index of the removed features in an array - layers_pruned. We store the index of the unremoved features in an array - layers_not_pruned. We then add the unpruned layers to the unpruned_layers array. We then add the activation layers in between and repeat the above method till we reach the last layer.

### One-shot Pruning (Reinitialization with Intialized Weights)

```python
def oneshot_pruning_reinit( post_training_model, pre_training_model, input_shape, output_shape, prune_ratio = 0.2):
    unpruned_layers = [] 
    layer_index = 0
    layers_pruned = []
    for name, param in post_training_model.named_parameters():
        if 'weight' in name:
            # Not pruning the last output layer
            if layer_index == len(post_training_model)-2:
                add_layer(unpruned_layers,
			input_shape,
			output_shape,
			get_data(post_training_model[layer_index].weight.data,
				param.data.shape,
				layers_pruned),
			nn.Sigmoid())
                continue
            # Sorting the features in a layer based on l1 norm
            param_with_skipped_input = pre_training_model[layer_index].weight.data[:, [col for col in range(param.data.shape[1]) if col not in layers_pruned]]
            sorted_layers = torch.linalg.norm(param_with_skipped_input, ord=1, dim=1).argsort(dim=-1)
            layers_not_pruned = sorted(sorted_layers[int(prune_ratio*param_with_skipped_input.shape[0]):])
            layers_pruned = sorted(sorted_layers[:int(prune_ratio*param_with_skipped_input.shape[0])])

            # Initialising unpruned neurons with pre-training values
            layer_data = param_with_skipped_input[layers_not_pruned, :] 
            add_layer(unpruned_layers, input_shape, layer_data.shape[0], layer_data)
            input_shape = layer_data.shape[0]
            #skipping every alternate relu layer
            layer_index=layer_index+2
    model = nn.Sequential(*unpruned_layers)
    index = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            param.data = unpruned_layers[index].data
            index=index+2
    return model
```

> This is similar to the above method, except here we re-initialize the model with the values of the pre-training model. (at line 18)
{: .prompt-tip }