'''
© 2024 Nokia
Licensed under the BSD 3-Clause Clear License
SPDX-License-Identifier: BSD-3-Clause-Clear
'''

import sys
import torch
import yaml
from utils import calcul_sinr, calcul_sinr_pred
from gnn import FastGNNLinearPrecodingLightning


# Set device
device = torch.device('cuda')

# Normalization coefficients
normalization_config = 'normalization_config.yaml'
with open(normalization_config, 'r') as config_file:
    normalization_dict = yaml.safe_load(config_file)
x_mean = torch.tensor(normalization_dict['x_mean'], device=device)
x_std = torch.tensor(normalization_dict['x_std'], device=device)
y_mean = torch.tensor(normalization_dict['y_mean'], device=device)
y_std = torch.tensor(normalization_dict['y_std'], device=device)

# Fixed rho_d parameter
rho_d = 314411439309.0463


def preprocess(G):
    n_aps, n_ues = G.shape

    # Preprocess data
    x = torch.reshape(G.T, (-1, 1))
    G_conj = torch.conj(G)
    G_inv = torch.inverse(G_conj.T @ G)
    G_dague = G_conj @ G_inv.T
    x1 = torch.reshape(G_dague.T, (-1, 1))
    # Apply logarithm
    x = torch.cat((torch.log2(x.abs()), x.angle(),
                    torch.log2(x1.abs()+1), x1.angle()), 1)
    # Input normalization
    x = (x - x_mean) / x_std

    # Compute edge index
    # Create the edges of the line graph structure where each node
    # represents a channel, i.e., a pair of UE and AP
    same_ap_edges = []
    same_ue_edges = []  # edges id from 0 to n_ues*n_aps-1
    # UE type edges
    for k in range(n_ues):
        for m1 in range(n_aps):
            for m2 in range(m1+1, n_aps):
                same_ue_edges.append([k*n_aps+m1, k*n_aps+m2])
                # reverse to make graph unoriented
                same_ue_edges.append([k*n_aps+m2, k*n_aps+m1])
    same_ue_edges = torch.tensor(same_ue_edges, dtype=torch.long)
    same_ue_edges = same_ue_edges.t().contiguous()
    # AP type edges
    for m in range(n_aps):
        for k1 in range(n_ues):
            for k2 in range(k1+1, n_ues):
                same_ap_edges.append([k1*n_aps+m, k2*n_aps+m])
                # reverse to make graph unoriented
                same_ap_edges.append([k2*n_aps+m, k1*n_aps+m])
    same_ap_edges = torch.tensor(same_ap_edges, dtype=torch.long)
    same_ap_edges = same_ap_edges.t().contiguous()

    return x, same_ap_edges, same_ue_edges, G_dague


def denormalize(y_hat):
    y_hat = y_hat * y_std + y_mean
    y_hat = torch.polar(torch.pow(2, y_hat[:, [0, 2, 4]]),
                        y_hat[:, [1, 3, 5]])
    y_hat = [y_hat[:, 0], y_hat[:, 1], y_hat[:, 2]-(1e-20)]
    return y_hat


# Possible values: 'highest', 'high', 'medium'
torch.set_float32_matmul_precision('medium')
torch.set_grad_enabled(False)

# Load model
model = FastGNNLinearPrecodingLightning.load_from_checkpoint(sys.argv[1]).model
model = model.to(device).eval()

# Load test data (one graph) and preprocess it
filename = sys.argv[2]
print('\nTesting OLP-GNN on file {}:'.format(filename))
G = torch.load(filename)
G = G.to(device)
n_aps, n_ues = G.shape
x, edge_index_ap, edge_index_ue, G_dague = preprocess(G)
x = x.to(device)
edge_index_ap = edge_index_ap.to(device)
edge_index_ue = edge_index_ue.to(device)

# Inference
y_hat = model(x, edge_index_ue, edge_index_ap)

# Denormalization
y_hat = denormalize(y_hat)

# Compute SINR
# With postprocessing
sinr_hat_postproc = calcul_sinr_pred(G.T, y_hat[0], y_hat[1], y_hat[2],
                            rho_d, n_aps, n_ues, G_dague.T)
# Without postprocessing
delta_nopostproc = y_hat[0] + y_hat[1] + y_hat[2]
sinr_hat_nopostproc = calcul_sinr(G.T, delta_nopostproc, rho_d, n_aps, n_ues)

print('------------------- With postprocessing -------------------') 
print('SINR (dB):', sinr_hat_postproc.tolist())
rel = (torch.mean(sinr_hat_postproc) - torch.min(sinr_hat_postproc)) \
    / torch.mean(sinr_hat_postproc)
print('(mean SINR - min SINR) / mean SINR =', rel.item())
print('std of SINR:', torch.std(sinr_hat_postproc).item())

print('------------------- No postprocessing -------------------') 
print('SINR (dB):', sinr_hat_nopostproc.tolist())
rel = (torch.mean(sinr_hat_nopostproc) - torch.min(sinr_hat_nopostproc)) \
    / torch.mean(sinr_hat_nopostproc)
print('(mean SINR - min SINR) / mean SINR =', rel.item())
print('std of SINR:', torch.std(sinr_hat_nopostproc).item())
