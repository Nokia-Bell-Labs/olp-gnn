'''
© 2024 Nokia
Licensed under the BSD 3-Clause Clear License
SPDX-License-Identifier: BSD-3-Clause-Clear
'''

import torch


# get_parameters function, called in deprocess_data function
# input:    preprocessed batch or individual graph
# output:   useful parameters of the batch or graph in appropriate format
def get_parameters(batch):

    n_aps, n_ues = batch['channel'].n_aps, batch['channel'].n_ues
    rho_d = batch['channel'].rho_d

    if isinstance(batch['channel'].n_aps, int):
        batch_size = 1
    else:
        batch_size = batch['channel'].n_aps.shape[0]

    if batch_size == 1:
        device = batch['channel'].x.device
        n_aps = torch.zeros(1, dtype=torch.int, device=device) + n_aps
        n_ues = torch.zeros(1, dtype=torch.int, device=device) + n_ues
        input_mean = batch['channel'].input_mean
        input_std = batch['channel'].input_std
        output_mean = batch['channel'].output_mean
        output_std = batch['channel'].output_std
        return input_mean, input_std, output_mean, output_std, n_aps, n_ues, \
            batch_size, rho_d

    input_mean = torch.repeat_interleave(
        batch['channel'].input_mean, batch['channel'].num_graph_node, dim=0)
    input_std = torch.repeat_interleave(
        batch['channel'].input_std, batch['channel'].num_graph_node, dim=0)
    output_mean = torch.repeat_interleave(
        batch['channel'].output_mean, batch['channel'].num_graph_node, dim=0)
    output_std = torch.repeat_interleave(
        batch['channel'].output_std, batch['channel'].num_graph_node, dim=0)

    return input_mean, input_std, output_mean, output_std, n_aps, n_ues, \
        batch_size, rho_d


def deprocess_data(batch, y_hat):

    input_mean, input_std, output_mean, output_std, n_aps, n_ues, \
      batch_size, rho_d = get_parameters(batch)

    # Deprocess the data
    G = batch['channel'].x*input_std+input_mean
    DeltaT = batch['channel'].y*output_std+output_mean
    y_hat = y_hat*output_std+output_mean

    # Combine the real features back to the original complex format
    G_dague = torch.polar((torch.pow(2, G[:, 2])-1), G[:, 3])
    G = torch.polar(torch.pow(2, G[:, 0]), G[:, 1])
    y_hat = torch.polar(torch.pow(2, y_hat[:, [0, 2, 4]]),
                        y_hat[:, [1, 3, 5]])
    y_hat = [y_hat[:, 0], y_hat[:, 1], y_hat[:, 2]-(1e-20)]
    DeltaT = torch.polar(torch.pow(2, DeltaT[:, [0, 2, 4]]),
                         DeltaT[:, [1, 3, 5]])
    DeltaT[:, 2] = DeltaT[:, 2] - (1e-20)
    DeltaT = torch.sum(DeltaT, dim=1, keepdim=True)

    return G, G_dague, DeltaT, y_hat, n_ues, n_aps, batch_size, rho_d


def power_control(Delta_hat, n_ues, n_aps):

    power = Delta_hat.clone()
    power = torch.linalg.norm(power, dim=1, keepdim=True)
    # indices where the power constraint is violated
    power_violated_index = power > 1
    # indices where the power constraint is satisfied
    power_ok_index = ~power_violated_index
    scaling_power = power_violated_index*power + power_ok_index
    scaling_power = scaling_power.expand(-1, n_ues)
    scaling_power = scaling_power.view(n_aps, n_ues)
    # scaling power tensor

    Delta_hat = Delta_hat/scaling_power
    return Delta_hat


def calcul_sinr(G, Delta, rho_d, n_aps, n_ues):

    G_T = G.view(n_ues, n_aps)
    Delta = Delta.view(n_ues, n_aps).T
    Delta = power_control(Delta, n_ues, n_aps)
    A = torch.matmul(G_T, Delta)
    A_diag = (torch.diag(A).abs())**2*rho_d
    A = rho_d*torch.linalg.norm(A, dim=1, keepdim=False)**2
    sinr = A_diag/(1+A-A_diag)

    return 10*torch.log10(sinr)


def calcul_sinr_pred(G, Delta1, Delta2, Delta3, rho_d, n_aps, n_ues, G_dague):

    G_T = G.view(n_ues, n_aps)
    G_dague = G_dague.view(n_ues, n_aps).T
    y1 = Delta1.view(n_ues, n_aps).T
    y2 = Delta2.view(n_ues, n_aps).T
    y3 = Delta3.view(n_ues, n_aps).T

    # Data transformation
    # impose A2 to have zero diagonal
    A2 = torch.matmul(G_T, y2)
    y2 = torch.matmul(G_dague, A2-torch.diag(torch.diag(A2)))
    # impose A1 to be diagonal and real
    A1 = torch.matmul(G_T, y1).real.to(torch.cfloat)
    A1 = torch.diag(torch.diag(A1))
    y1 = torch.matmul(G_dague, A1)

    Delta = (y1+y2+y3).T
    return calcul_sinr(G, Delta, rho_d, n_aps, n_ues)


def get_sinr(batch, y_hat):
    device = y_hat.device

    G, G_dague, y, y_hat, n_ues, n_aps, batch_size, rho_d = deprocess_data(
        batch, y_hat)

    if batch_size == 1:
        idx = [0, batch['channel'].num_graph_node]
    else:
        idx = torch.cat((torch.zeros(1, dtype=torch.int, device=device),
                         torch.cumsum(batch['channel'].num_graph_node, 0)),
                        0)
    SINR_hat = torch.cat([
        calcul_sinr_pred(G[idx[i]:idx[i+1]], y_hat[0][idx[i]:idx[i+1]],
                         y_hat[1][idx[i]:idx[i+1]],
                         y_hat[2][idx[i]:idx[i+1]],
                         rho_d[i], n_aps[i], n_ues[i],
                         G_dague[idx[i]:idx[i+1]])
        for i in range(len(idx)-1)], 0)
    SINR = torch.cat([calcul_sinr(G[idx[i]:idx[i+1]], y[idx[i]:idx[i+1]],
                                  rho_d[i], n_aps[i], n_ues[i])
                      for i in range(len(idx)-1)], 0)
    return SINR, SINR_hat
