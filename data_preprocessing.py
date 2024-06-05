'''
© 2024 Nokia
Licensed under the BSD 3-Clause Clear License
SPDX-License-Identifier: BSD-3-Clause-Clear
'''

import numpy as np
import os
import sys
import torch
import yaml
from torch_geometric.data import HeteroData


def parse_raw_data_file(data_path, apply_log=True):

    # Load raw data file
    raw_data = np.load(data_path)
    list_sinr = raw_data['SINR']
    list_delta = raw_data['Delta']
    list_G = raw_data['G']
    list_A = raw_data['A']

    n = len(list_sinr)
    n_aps, n_ues = list_delta[0].shape
    SINR = torch.zeros((n, 1), dtype=torch.float)
    X = torch.zeros((n, n_aps*n_ues, 4), dtype=torch.float)
    Y = torch.zeros((n, n_aps*n_ues, 6), dtype=torch.float)

    # Iterate over each graph and preprocess them
    for i in range(n):
        G = torch.from_numpy(list_G[i])
        Delta = torch.from_numpy(list_delta[i])
        A = torch.from_numpy(list_A[i])

        sinr = list_sinr[i]
        x = torch.reshape(G.T, (-1, 1))
        G_conj = torch.conj(G)
        G_inv = torch.inverse(G_conj.T @ G)
        G_dague = G_conj @ G_inv.T
        x1 = torch.reshape(G_dague.T, (-1, 1))

        A_diag = torch.diag(torch.diag(A))
        y1 = G_dague @ (A_diag)
        y2 = G_dague @ (A - A_diag)
        y3 = Delta - (G_dague @ A) + 1e-20

        y1 = torch.reshape(y1.T, (-1, 1))
        y2 = torch.reshape(y2.T, (-1, 1))
        y3 = torch.reshape(y3.T, (-1, 1))

        if apply_log:
            x = torch.cat((torch.log2(x.abs()), x.angle(),
                           torch.log2(x1.abs()+1), x1.angle()), 1)
            y = torch.cat((torch.log2(y1.abs()), y1.angle(),
                           torch.log2(y2.abs()), y2.angle(),
                           torch.log2(y3.abs()), y3.angle()), 1)
        else:
            x = torch.cat((x.abs(), x.angle(), x1.abs(), x1.angle()), 1)
            y = torch.cat((y1.abs(), y1.angle(), y2.abs(), y2.angle(),
                           y3.abs(), y3.angle()), 1)

        X[i] = x
        Y[i] = y
        SINR[i] = sinr

    return X, Y, n, n_aps, n_ues, SINR


def create_graph_dataset(files_info, normalization_dict=None, apply_log=True):
    graphs_data = {}
    dataset = []
    X_tot = []
    Y_tot = []
    files_split_dict = {}

    if normalization_dict is None:
        # Get the statistics over the dataset
        for file_data_path in files_info.keys():
            X, Y, n_samples, n_aps, n_ues, SINR = \
                parse_raw_data_file(file_data_path, apply_log)
            X_tot.append(X.reshape((-1, 4)))
            Y_tot.append(Y.reshape((-1, 6)))
        X_tot = torch.cat(X_tot)
        Y_tot = torch.cat(Y_tot)
        x_mean, x_std = torch.mean(X_tot, dim=0), torch.std(X_tot, dim=0)
        y_mean, y_std = torch.mean(Y_tot, dim=0), torch.std(Y_tot, dim=0)
        normalization_dict = {'x_mean': x_mean.tolist(),
                              'x_std': x_std.tolist(),
                              'y_mean': y_mean.tolist(),
                              'y_std': y_std.tolist()}
    else:
        # If the normalization statistics are provided
        x_mean = torch.tensor(normalization_dict['x_mean'])
        x_std = torch.tensor(normalization_dict['x_std'])
        y_mean = torch.tensor(normalization_dict['y_mean'])
        y_std = torch.tensor(normalization_dict['y_std'])

    # Iterate over each sample of each scenario and save each sample as a graph
    # pytorch geometric structure
    for file_data_path in files_info.keys():
        file_name, file_split_info, rho_d = files_info[file_data_path]
        files_split_dict[file_name] = file_split_info
        dataset = []
        X, Y, n_samples, n_aps, n_ues, SINR = \
            parse_raw_data_file(file_data_path, apply_log)
        print(('Preprocessing file {} with {} samples'
               ' (train={}, val={}, test={})')
              .format(file_name, n_samples,
                      int(n_samples*file_split_info[0]),
                      int(n_samples*file_split_info[1]),
                      int(n_samples*file_split_info[2])))

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

        # Create and save the pytorch geometric graphs
        for i in range(n_samples):

            x = X[i]
            y = Y[i]
            sinr = SINR[i]

            x = (x-x_mean)/x_std
            y = (y-y_mean)/y_std

            data = HeteroData()
            data['channel'].x = x
            data['channel'].y = y
            data['channel', 'same_ue', 'channel'].edge_index = same_ue_edges
            data['channel', 'same_ap', 'channel'].edge_index = same_ap_edges

            # add metadata
            data['channel'].sinr = sinr
            data['channel'].input_mean = torch.reshape(x_mean, (1, 4))
            data['channel'].input_std = torch.reshape(x_std, (1, 4))
            data['channel'].output_mean = torch.reshape(y_mean, (1, 6))
            data['channel'].output_std = torch.reshape(y_std, (1, 6))
            data['channel'].n_ues = n_ues
            data['channel'].n_aps = n_aps
            data['channel'].num_graph_node = n_ues*n_aps
            data['channel'].rho_d = torch.tensor(rho_d, dtype=torch.double)

            dataset.append(data)

        graphs_data[file_name] = dataset

    return graphs_data, files_split_dict, normalization_dict


if __name__ == "__main__":
    data_path = sys.argv[1]
    save_filename = 'dataset_train.pt'

    normalization_config = 'normalization_config.yaml'
    normalization_dict = None
    if os.path.exists(normalization_config):
        with open(normalization_config, 'r') as config_file:
            normalization_dict = yaml.safe_load(config_file)
        print('{} file found!'.format(normalization_config))

    test_files = ['data_olp_rural_24_4.npz',
                  'data_olp_rural_24_5.npz',
                  'data_olp_rural_24_6.npz',
                  'data_olp_rural_24_9.npz',
                  'data_olp_rural_32_4.npz',
                  'data_olp_rural_32_6.npz',
                  'data_olp_rural_32_8.npz',
                  'data_olp_rural_32_9.npz',
                  'data_olp_rural_32_12.npz',
                  'data_olp_rural_32_16.npz',
                  'data_olp_rural_48_8.npz',
                  'data_olp_rural_48_12.npz',
                  'data_olp_rural_48_16.npz',
                  'data_olp_rural_48_24.npz',
                  'data_olp_rural_64_6.npz',
                  'data_olp_rural_64_9.npz',
                  'data_olp_rural_64_12.npz',
                  'data_olp_rural_64_18.npz',
                  'data_olp_rural_64_24.npz',
                  'data_olp_rural_64_32.npz',
                  'data_olp_rural_96_9.npz',
                  'data_olp_rural_96_18.npz',
                  'data_olp_rural_96_27.npz',
                  'data_olp_rural_96_36.npz',
                  ]
    test_splits = [(0, 0, 1.0)] * len(test_files)
    val_files = ['data_olp_urban_24_4.npz', 'data_los_60GHz_olp_24_4.npz',
                 'data_olp_urban_24_5.npz', 'data_los_60GHz_olp_24_5.npz',
                 'data_olp_urban_24_6.npz', 'data_los_60GHz_olp_24_6.npz',
                 'data_olp_urban_24_9.npz', 'data_los_60GHz_olp_24_9.npz',
                 'data_olp_urban_32_4.npz', 'data_los_60GHz_olp_32_4.npz',
                 'data_olp_urban_32_8.npz',  'data_los_60GHz_olp_32_8.npz',
                 'data_olp_urban_32_12.npz', 'data_los_60GHz_olp_32_12.npz',
                 'data_olp_urban_32_16.npz', 'data_los_60GHz_olp_32_16.npz',
                 'data_olp_urban_48_8.npz',  'data_los_60GHz_olp_48_8.npz',
                 'data_olp_urban_48_12.npz',  'data_los_60GHz_olp_48_12.npz',
                 'data_olp_urban_48_16.npz', 'data_los_60GHz_olp_48_16.npz',
                 'data_olp_urban_48_24.npz',  'data_los_60GHz_olp_48_24.npz',
                 'data_olp_urban_64_6.npz',  'data_los_60GHz_olp_64_6.npz',
                 'data_olp_urban_64_12.npz',  'data_los_60GHz_olp_64_12.npz',
                 'data_olp_urban_64_24.npz',  'data_los_60GHz_olp_64_24.npz',
                 'data_los_60GHz_olp_64_32.npz',
                 'data_olp_urban_96_9.npz', 'data_los_60GHz_olp_96_9.npz',
                 'data_olp_urban_96_18.npz', 'data_los_60GHz_olp_96_18.npz',
                 'data_olp_urban_96_27.npz', 'data_los_60GHz_olp_96_27.npz',
                 'data_olp_urban_96_36.npz', 'data_los_60GHz_olp_96_36.npz',
                 ]
    val_splits = [(0, 0.5, 0.5)] * len(val_files)
    val_files.append('data_olp_urban_64_32.npz')
    val_splits.append((0, 0.05, 0.05))
    train_files = ['data_olp_urban_32_6.npz', 'data_los_60GHz_olp_32_6.npz',
                   'data_olp_urban_32_9.npz', 'data_los_60GHz_olp_32_9.npz',
                   'data_olp_urban_64_9.npz', 'data_los_60GHz_olp_64_9.npz',
                   'data_olp_urban_64_18.npz', 'data_los_60GHz_olp_64_18.npz'
                   ]
    train_splits = [(0.9, 0.05, 0.05)] * len(train_files)
    files = test_files + val_files + train_files
    splits = test_splits + val_splits + train_splits

    # TODO: this parameter is hard-coded. All our datasets share
    # the same rho_d value, see data_generation.
    rho_d = 314411439309.0463

    files_info = {}
    for filename, split in zip(files, splits):
        basename = os.path.splitext(filename)[0]
        files_info[os.path.join(data_path, filename)] = (basename, split, rho_d)

    dataset, files_split_dict, normalization_dict = \
        create_graph_dataset(files_info, normalization_dict)

    # Create normalization_config file
    if not os.path.exists(normalization_config):
        with open(normalization_config, 'w') as config_file:
            yaml.dump(normalization_dict, config_file,
                      default_flow_style=False)
        print('new {} file created!'.format(normalization_config))

    # Create dataset file
    torch.save((dataset, files_split_dict), save_filename)
