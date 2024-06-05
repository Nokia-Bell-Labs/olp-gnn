'''
© 2024 Nokia
Licensed under the BSD 3-Clause Clear License
SPDX-License-Identifier: BSD-3-Clause-Clear
'''

from gnn import FastGNNLinearPrecodingLightning
import torch
import sys
import os


# Possible values: 'highest', 'high', 'medium'
torch.set_float32_matmul_precision('medium')

device = torch.device('cuda')

n_warmups = 1
n_repetitions = 10

lightning_model = \
    FastGNNLinearPrecodingLightning.load_from_checkpoint(sys.argv[1])

model = lightning_model.model

datasets, datasets_split_dict = torch.load('dataset_train.pt')
runtimes = {}

fig_dir = sys.argv[2]
save_text_file = os.path.join(fig_dir, "runtime_results.txt")

for filename in datasets_split_dict.keys():
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

    # Compile model for faster runtimes
    compiled_model = torch.compile(model, mode="reduce-overhead")
    compiled_model = compiled_model.to(device).eval()

    scen_runtimes = []
    for run_idx in range(n_warmups + n_repetitions):
        torch.cuda.empty_cache()
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

            start = torch.cuda.Event(enable_timing=True)
            end_preprocess = torch.cuda.Event(enable_timing=True)
            end_inference = torch.cuda.Event(enable_timing=True)
            end_postprocess = torch.cuda.Event(enable_timing=True)
            start.record()

            # Preprocess
            G_inv = torch.linalg.inv(torch.matmul(torch.conj(G).T, G))
            G_dague = torch.matmul(torch.conj(G), G_inv.T)
            G = torch.reshape(G.T, (-1, 1))
            G_dague = torch.reshape(G_dague.T, (-1, 1))
            x = torch.cat((torch.log2(G.abs()), G.angle(),
                           torch.log2(G_dague.abs()+1), G_dague.angle()), 1)
            x = (x - input_mean) / input_std
            end_preprocess.record()

            output = compiled_model(x, edge_index_ue, edge_index_ap)

            end_inference.record()

            # Postprocess: compute the power control coefficients delta from
            # the GNN output
            delta = output*output_std+output_mean
            delta = torch.polar(torch.pow(2, delta[:, [0, 2, 4]]),
                                delta[:, [1, 3, 5]])
            delta = delta[:, 0]+delta[:, 1]+delta[:, 2]-(1e-20)
            end_postprocess.record()

            torch.cuda.synchronize()
            elapsed1 = start.elapsed_time(end_preprocess)
            elapsed2 = end_preprocess.elapsed_time(end_inference)
            elapsed3 = end_inference.elapsed_time(end_postprocess)
            if run_idx >= n_warmups:
                scen_runtimes.append([elapsed1, elapsed2, elapsed3])

    scen_runtimes = torch.as_tensor(scen_runtimes)
    runtimes[filename] = scen_runtimes
    names = ['preprocessing', 'inference', 'postprocessing']
    with open(save_text_file, "a") as f:
        print('\nTesting scenario {}:'.format(filename), file=f)
        for idx, name in enumerate(names):
            tmp = scen_runtimes[:, idx]
            text = ('{} runtimes (ms): mean={:.2e}, std={:.2e},'
                    ' min={:.2e}, max={:.2e}').format(
                        name, torch.mean(tmp), torch.std(tmp),
                        torch.min(tmp), torch.max(tmp))
            print(text)
            print(text, file=f)
        total = torch.sum(scen_runtimes, 1)
        text = ('Total runtimes (ms): mean={:.2e}, std={:.2e},'
                ' min={:.2e}, max={:.2e}').format(
                    torch.mean(total), torch.std(total),
                    torch.min(total), torch.max(total))
        print(text)
        print(text, file=f)
