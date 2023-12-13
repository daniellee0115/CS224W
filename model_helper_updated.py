# Note 1: Parameter has been adjusted to be compatible with a batch graph training
# Note 2: Another layer has been added to perform k-hop (k = 2)

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
    - The detailed mathmatics for GAT is described at
        https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GATv2Conv.html
        https://docs.dgl.ai/en/0.8.x/tutorials/models/1_gnn/9_gat.html
        Every steps are performed at the backend over forward function of GATv2Conv

'''
class GATv2(nn.Module):
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

    def forward(self, g, inputs):
        h = inputs
        for l in range(self.num_layers):
            h = self.gatv2_layers[l](g, h).flatten(1)
        # output projection
        score = self.gatv2_layers[-1](g, h).mean(1)
        return score
    
    ### GAT2Conv testing ###
    args = {
        'num_layers': 2,
        'input_dim': 7,
        'num_hidden': 64, # hidden layer unit number
        'num_heads': 2,
        'out_dim': 2,
        'in_drop': 0.3, # default: 0.7
        'attn_drop': 0.5, # default: 0.7 
        'negative_slope': 0.2, # default: 0.2
        'residual': False # default
    }
    heads = ([args['num_heads']] * args['num_layers']) + [1]


'''
class GATLayer
Below is the step-by-step GAT model implementation described at https://docs.dgl.ai/en/0.8.x/tutorials/models/1_gnn/9_gat.html

Note
    - This was part of our original implementation of GAT
    - But this implementation doesn't have a way to do input or attention drop, so we switched over the GATv2Conv
'''

class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GATLayer, self).__init__()
        # self.g = g
        # equation (1)
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        # equation (2)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

    def edge_attention(self, edges):
        # edge UDF for equation (2)
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        # message UDF for equation (3) & (4)
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        # reduce UDF for equation (3) & (4)
        # equation (3)
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        # equation (4)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

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
    def __init__(self, in_dim, out_dim, num_heads, merge='cat'):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATLayer(in_dim, out_dim))
        self.merge = merge

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
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads, device='cuda'):
        super(GAT, self).__init__()
        self.layer1 = MultiHeadGATLayer(in_dim, hidden_dim, num_heads)
        self.layer2 = MultiHeadGATLayer(hidden_dim * num_heads, hidden_dim, num_heads)
        self.layer3 = MultiHeadGATLayer(hidden_dim * num_heads, out_dim, 1)
        self.device = device

    def forward(self, g, h):
        g = g.to(self.device)
        h = self.layer1(g, h)
        h = F.elu(h)
        h = self.layer2(g, h)
        h = F.elu(h)
        h = self.layer3(g, h)
        return h