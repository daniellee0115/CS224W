'''
model_helper_updated.py
Containing the model of Graph Attention Network as a CS224W project By Kaya Guvendi, Daniel Lee, Gyu Kim
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GATv2Conv

'''
GATv2 is the wrapper for GATv2Conv to easily generate multiple layers of GATv2Conv models
It contains additional features
    @ num_layers: The number of total layers
    @ in_dim: Input feature dimension
    @ num_hidden: hidden layer feature dimension
    @ heads: the list of number of heads for each layer to perform multihead GAT (last layer should be 1)
    @ Other variables are the same as described in GATv2Conv description page

Note
    - Learnable weight initialization steps are included as part of GATv2Conv init.
    - Used GATv2Conv as a backbone GAT implementation. The detailed mathmatics for GAT is described at
        https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GATv2Conv.html
        https://docs.dgl.ai/en/0.8.x/tutorials/models/1_gnn/9_gat.html
'''
class GATv2(nn.Module):
    # Initialize GATv2 using the given hyperparameters, see class header for their values
    def __init__(
        self,
        num_layers,
        in_dim,
        num_hidden,
        num_classes,
        heads,
        activation,
        feat_drop,
        attn_drop,
        negative_slope,
        residual,
    ):
        super(GATv2, self).__init__()
        self.num_layers = num_layers
        self.gatv2_layers = nn.ModuleList()
        self.activation = activation
        # input projection (no residual)
        self.gatv2_layers.append(
            GATv2Conv(
                in_dim,
                num_hidden,
                heads[0],
                feat_drop,
                attn_drop,
                negative_slope,
                False,
                self.activation,
                bias=False,
                share_weights=True,
            )
        )
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gatv2_layers.append(
                GATv2Conv(
                    num_hidden * heads[l - 1],
                    num_hidden,
                    heads[l],
                    feat_drop,
                    attn_drop,
                    negative_slope,
                    residual,
                    self.activation,
                    bias=False,
                    share_weights=True,
                )
            )
        # output projection
        self.gatv2_layers.append(
            GATv2Conv(
                num_hidden * heads[-2],
                num_classes,
                heads[-1],
                feat_drop,
                attn_drop,
                negative_slope,
                residual,
                None,
                bias=False,
                share_weights=True,
            )
        )

    # Performs a forward pass through the layers of this GAT
    def forward(self, g, inputs):
        h = inputs
        for l in range(self.num_layers):
            h = self.gatv2_layers[l](g, h).flatten(1)
        # output projection
        score = self.gatv2_layers[-1](g, h).mean(1)
        return score
    
'''
class GATLayer
Below is the step-by-step GAT model implementation described at https://docs.dgl.ai/en/0.8.x/tutorials/models/1_gnn/9_gat.html

GAT step-by-step
    equation (1): Linear transformation of the lower layer embedding using learnable weight matrix
    equation (2): Computing a pair-wise un-normalized attention score between two neighbors and applying leakyReLu
    equation (3): Applying a softmax to normalize the attention scores on each node’s incoming edges
    equation (4): The embeddings from neighbors are aggregated together, scaled by the attention scores

Note
    - This was part of our original implementation of GAT
    - But this implementation doesn't have a way to do input or attention drop, so we switched over the GATv2Conv

'''
class GATLayer(nn.Module):
    # Initializes the GAT layer with the given parameters, which correspond to the input and output dimensions
    def __init__(self, in_dim, out_dim):
        super(GATLayer, self).__init__()
        # self.g = g
        # equation (1)
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        # equation (2)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
        self.reset_parameters()

    # Resets the parameters of the GAT using Xavier Normal initializations
    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

    # Computes attention values for the given edges
    def edge_attention(self, edges):
        # edge UDF for equation (2)
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}

    # Formats the messages created given the edges 
    def message_func(self, edges):
        # message UDF for equation (3) & (4)
        return {'z': edges.src['z'], 'e': edges.data['e']}

    # Performs the aggregation for the GAT layer
    def reduce_func(self, nodes):
        # reduce UDF for equation (3) & (4)
        # equation (3)
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        # equation (4)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    # Performs a forward pass through the GAT layer given a graph g and node embeddings h
    def forward(self, g, h):
        # equation (1)
        z = self.fc(h)
        g.ndata['z'] = z
        # equation (2)
        g.apply_edges(self.edge_attention)
        # equation (3) & (4)
        g.update_all(self.message_func, self.reduce_func)
        return g.ndata.pop('h')

'''
class MultiHeadGATLayer
From https://docs.dgl.ai/en/0.8.x/tutorials/models/1_gnn/9_gat.html
Implementation of Multihead GAT Layer by using GATLayer defined above

Note
    - This was part of our original implementation of GAT
    - But this implementation doesn't have a way to do input or attention drop, so we switched over the GATv2Conv
'''   
class MultiHeadGATLayer(nn.Module):
    # Initializes a multi-headed GAT layer operating with the number of given heads, input dimensions, and outputs dimension
    # Defaults to concatenation of different heads' outputs, but can be made to mean them instead
    def __init__(self, in_dim, out_dim, num_heads, merge='cat'):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATLayer(in_dim, out_dim))
        self.merge = merge

    # Performs a forward pass through this GAT layer
    def forward(self, g, h):
        head_outs = [attn_head(g, h) for attn_head in self.heads]
        if self.merge == 'cat':
            # concat on the output feature dimension (dim=1)
            return torch.cat(head_outs, dim=1)
        else:
            # merge using average
            return torch.mean(torch.stack(head_outs))

'''
class GAT
From https://docs.dgl.ai/en/0.8.x/tutorials/models/1_gnn/9_gat.html
Implementation of Multi-layered Multihead GAT Layer with adjustable parameters for hyperparameter tuning

Note
    - This was part of our original implementation of GAT
    - But this implementation doesn't have a way to do input or attention drop, so we switched over the GATv2Conv
'''

class GAT(nn.Module):
    # Initializes a 3-layer multi-headed GAT with the given number of heads on the given device.
    # Takes in inputs of in_dim dimensions and return out_dim-dimensional outputs while the intermediate
    # values are kept at hidden_dim dimensions.
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads, device='cuda'):
        super(GAT, self).__init__()
        self.layer1 = MultiHeadGATLayer(in_dim, hidden_dim, num_heads) # Input layer
        self.layer2 = MultiHeadGATLayer(hidden_dim * num_heads, hidden_dim, num_heads) # Hidden layer
        self.layer3 = MultiHeadGATLayer(hidden_dim * num_heads, out_dim, 1) # Output layer
        self.device = device

    # Performs a forward pass through this GAT
    def forward(self, g, h):
        g = g.to(self.device)
        h = self.layer1(g, h)
        h = F.elu(h)
        h = self.layer2(g, h)
        h = F.elu(h)
        h = self.layer3(g, h)
        return h
