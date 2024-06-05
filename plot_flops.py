'''
© 2024 Nokia
Licensed under the BSD 3-Clause Clear License
SPDX-License-Identifier: BSD-3-Clause-Clear
'''

from gnn import FastGNNLinearPrecodingLightning
from pypapi import events, papi_high
import matplotlib.pyplot as plt
import numpy as np
import torch
import sys
import os


plt.set_loglevel("error")  # Suppress warnings

# Possible values: 'highest', 'high', 'medium'
torch.set_float32_matmul_precision('highest')

device = torch.device('cpu')

papi_events = [events.PAPI_SP_OPS]

lightning_model = \
    FastGNNLinearPrecodingLightning.load_from_checkpoint(sys.argv[1])

model = lightning_model.model

datasets, datasets_split_dict = torch.load('dataset_train.pt')

considered_datasets = [
    'data_olp_rural_24_4',
    'data_olp_rural_24_5',
    'data_olp_rural_24_6',
    'data_olp_rural_24_9',
    'data_olp_rural_32_4',
    'data_olp_rural_32_6',
    'data_olp_rural_32_8',
    'data_olp_rural_32_9',
    'data_olp_rural_32_12',
    'data_olp_rural_32_16',
    'data_olp_rural_48_8',
    'data_olp_rural_48_12',
    'data_olp_rural_48_16',
    'data_olp_rural_48_24',
    'data_olp_rural_64_6',
    'data_olp_rural_64_9',
    'data_olp_rural_64_12',
    'data_olp_rural_64_18',
    'data_olp_rural_64_24',
    'data_olp_rural_64_32',
    'data_olp_rural_96_9',
    'data_olp_rural_96_18',
    'data_olp_rural_96_27',
    'data_olp_rural_96_36'
    ]

raw_socp_dir = sys.argv[2]
raw_socp_files = {}
for filename in considered_datasets:
    raw_socp_files[filename] = os.path.join(raw_socp_dir, filename+'.npz')

fig_dir = sys.argv[3]
save_text_file = os.path.join(fig_dir, "flops_results.txt")

list_gnn_flops = []
list_socp_flops = []
list_n_edges = []

for filename in datasets_split_dict.keys():
    if filename not in considered_datasets:
        continue
    print('\nTesting scenario {}:'.format(filename))

    # Get testing dataset
    dataset = datasets[filename]
    split = datasets_split_dict[filename]
    test_dataset = dataset[int((split[0]+split[1]) * len(dataset)):
                           int((split[0]+split[1]+split[2]) * len(dataset))]

    # Get preprocessing normalization statistics
    input_mean = test_dataset[0]['channel'].input_mean
    input_std = test_dataset[0]['channel'].input_std
    output_mean = test_dataset[0]['channel'].output_mean
    output_std = test_dataset[0]['channel'].output_std
    input_mean = input_mean.to(device)
    input_std = input_std.to(device)
    output_mean = output_mean.to(device)
    output_std = output_std.to(device)

    model = model.to(device).eval()

    dataset_flops_gnn = []
    for graph in test_dataset:
        # Transfer data to device
        x = graph['channel'].x
        x = x.to(device)
        edge_index_ue = graph['channel', 'same_ue', 'channel'].edge_index
        edge_index_ue = edge_index_ue.to(device)
        edge_index_ap = graph['channel', 'same_ap', 'channel'].edge_index
        edge_index_ap = edge_index_ap.to(device)
        n_ues = graph['channel'].n_ues
        n_aps = graph['channel'].n_aps

        # Deprocess so that we can account for the preprocessing time below
        deproc_inputs = x*input_std+input_mean
        G = torch.polar(torch.pow(2, deproc_inputs[:, 0]),
                        deproc_inputs[:, 1])
        G = G.reshape((n_ues, n_aps)).T

        # Start flops counter
        papi_high.start_counters(papi_events)

        # Preprocess
        G_inv = torch.linalg.inv(torch.matmul(torch.conj(G).T, G))
        G_dague = torch.matmul(torch.conj(G), G_inv.T)
        G = torch.reshape(G.T, (-1, 1))
        G_dague = torch.reshape(G_dague.T, (-1, 1))
        x = torch.cat((torch.log2(G.abs()), G.angle(),
                       torch.log2(G_dague.abs()+1), G_dague.angle()), 1)
        x = (x - input_mean) / input_std

        # GNN inference
        output = model(x, edge_index_ue, edge_index_ap)

        # Postprocess: compute the power control coefficients delta from
        # the GNN output
        delta = output*output_std+output_mean
        delta = torch.polar(torch.pow(2, delta[:, [0, 2, 4]]),
                            delta[:, [1, 3, 5]])
        delta = delta[:, 0]+delta[:, 1]+delta[:, 2]-(1e-20)

        flops = papi_high.stop_counters()
        dataset_flops_gnn.append(flops[0])

    n_edges = n_aps * n_ues * (n_aps + n_ues - 2.0)
    list_n_edges.append(n_edges)

    list_gnn_flops.append(np.mean(dataset_flops_gnn))

    # Load file with raw B-SOCP results containing their flops count
    socp_flops = np.load(raw_socp_files[filename])['flops']
    list_socp_flops.append(np.mean(socp_flops))

    # Save results in text file
    text1 = 'B-SOCP FLOPs: mean={:.2e}, std={:.2e}'.format(
        np.mean(socp_flops), np.std(socp_flops))
    text2 = 'GNN FLOPs: mean={:.2e}, std={:.2e}'.format(
        np.mean(dataset_flops_gnn), np.std(dataset_flops_gnn))
    text3 = 'B-SOCP / GNN FLOPs ratio: {}'.format(
        np.mean(socp_flops)/np.mean(dataset_flops_gnn))
    print(text1)
    print(text2)
    print(text3)
    with open(save_text_file, "a") as f:
        print('\nTesting scenario {}:'.format(filename), file=f)
        print(text1, file=f)
        print(text2, file=f)
        print(text3, file=f)

# Plot figure
plt.figure(figsize=(7, 4))
plt.ticklabel_format(axis='both', style='sci', scilimits=(0, 0))
plt.scatter(list_n_edges, list_socp_flops, marker='o', facecolors='none',
            edgecolors='#1f77b4', label='B-SOCP')
plt.scatter(list_n_edges, list_gnn_flops, marker='x',
            facecolors='#ff7f0e', label='OLP-GNN')
ax = plt.gca()
ax.set_yscale('log')
plt.ylabel('FLOPs count')
plt.xlabel('Number of edges $MK(M+K-2)$')
plt.grid()
plt.legend()
plt.savefig(os.path.join(fig_dir, "FLOPs_count.eps"), format='eps')
plt.close()
