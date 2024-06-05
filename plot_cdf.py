'''
© 2024 Nokia
Licensed under the BSD 3-Clause Clear License
SPDX-License-Identifier: BSD-3-Clause-Clear
'''

import os
import sys
import torch
import numpy as np
from utils import get_sinr
from gnn import FastGNNLinearPrecodingLightning, GNNDataModule
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import cycle


def _cdf_loss(gt, pred):
    median_gt = np.median(gt)
    median_loss = 100 * (median_gt - np.median(pred)) / median_gt
    centile5_gt = np.percentile(gt, 5)
    centile5_loss = 100 * (centile5_gt - np.percentile(pred, 5)) / centile5_gt
    return median_loss, centile5_loss


plt.set_loglevel("error")  # Suppress warnings

fig_dir = sys.argv[3]

# Possible values: 'highest', 'high', 'medium'
torch.set_float32_matmul_precision('medium')
torch.set_grad_enabled(False)

device = torch.device('cuda')

model = FastGNNLinearPrecodingLightning.load_from_checkpoint(sys.argv[1])
model = model.to(device).eval()

train_bs = 1
val_bs = 1
test_bs = 8
dm = GNNDataModule('dataset_train.pt', train_bs, val_bs, test_bs)
dm.setup(stage='test')
datasets = dm.test_dataloader()

data_dir = sys.argv[2]
mr_zf_datasets = {'data_olp_rural_32_16':
                  (os.path.join(data_dir, 'data_mr_rural_32_16.npz'),
                   os.path.join(data_dir, 'data_zf_rural_32_16.npz')),
                  'data_olp_rural_96_36':
                  (os.path.join(data_dir, 'data_mr_rural_96_36.npz'),
                   os.path.join(data_dir, 'data_zf_rural_96_36.npz')),
                  'data_olp_urban_96_36':
                  (os.path.join(data_dir, 'data_mr_urban_96_36.npz'),
                   os.path.join(data_dir, 'data_zf_urban_96_36.npz')),
                  'data_los_60GHz_olp_96_36':
                  (os.path.join(data_dir, 'data_los_60GHz_mr_96_36.npz'),
                   os.path.join(data_dir, 'data_los_60GHz_zf_96_36.npz'))
                  }

all_n_aps = np.zeros(len(dm.filenames))
all_n_ues = np.zeros(len(dm.filenames))
se_median_losses = np.zeros(len(dm.filenames))
se_95likely_losses = np.zeros(len(dm.filenames))

for dataset_idx, filename in enumerate(dm.filenames):
    print('\nTesting scenario {}:'.format(filename))

    # Get testing dataset
    dataset = datasets[dataset_idx]

    n_ues = next(iter(dataset))['channel'].n_ues[0].item()
    n_aps = next(iter(dataset))['channel'].n_aps[0].item()
    all_n_aps[dataset_idx] = n_aps
    all_n_ues[dataset_idx] = n_ues
    sinrs = []
    sinrs_hat = []

    for batch_idx, batch in enumerate(dataset):
        batch = batch.to(device)
        y_hat = model(batch)
        sinr, sinr_hat = get_sinr(batch, y_hat)
        sinrs.extend(torch.split(sinr, n_ues))
        sinrs_hat.extend(torch.split(sinr_hat, n_ues))

    sinrs = torch.stack(sinrs).numpy(force=True)
    sinrs_hat = torch.stack(sinrs_hat).numpy(force=True)

    se = np.log2(1+10**(sinrs.flatten()/10))
    se_hat = np.log2(1+10**(sinrs_hat.flatten()/10))
    l1, l2 = _cdf_loss(se, se_hat)
    se_median_losses[dataset_idx] = l1
    se_95likely_losses[dataset_idx] = l2
    print('se cdf loss: median={:.2f}%, 95-likely={:.2f}%'.format(l1, l2))

    if filename in mr_zf_datasets:
        mr_file, zf_file = mr_zf_datasets[filename]
        mr_min_sinrs = np.load(mr_file)['SINR']
        mr_se = np.log2(1+10**(mr_min_sinrs/10))
        zf_min_sinrs = np.load(zf_file)['SINR']
        zf_se = np.log2(1+10**(zf_min_sinrs/10))
        mr1, mr2 = _cdf_loss(mr_se, se_hat)
        print('MR-to-GNN se cdf loss: median={:.2f}%, 95-likely={:.2f}%'
              .format(mr1, mr2))
        zf1, zf2 = _cdf_loss(zf_se, se_hat)
        print('ZF-to-GNN se cdf loss: median={:.2f}%, 95-likely={:.2f}%'
              .format(zf1, zf2))

    plt.figure(figsize=(4, 3.2))
    sns.ecdfplot(se, label="OLP", linestyle='-')
    sns.ecdfplot(se_hat, label="OLP-GNN", linestyle=':', marker='x',
                 markevery=0.05)
    if filename in mr_zf_datasets:
        sns.ecdfplot(mr_se, label="MR", linestyle='-.')
        sns.ecdfplot(zf_se, label="ZF", linestyle='--')
    plt.xlabel('Spectral efficiency (bit/s/Hz)')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "se_cdf_"+filename+".eps"), format='eps')
    plt.close()


# Save SE CDF results in text file
se_cdf_txt_file = os.path.join(fig_dir, "se_cdf_results.txt")
with open(se_cdf_txt_file, "w") as text_file:
    for dataset_idx, filename in enumerate(dm.filenames):
        print("{}: median={:.2f}%, 95-likely={:.2f}%".format(
            filename, se_median_losses[dataset_idx],
            se_95likely_losses[dataset_idx]), file=text_file)
