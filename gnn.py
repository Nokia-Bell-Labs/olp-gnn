'''
© 2024 Nokia
Licensed under the BSD 3-Clause Clear License
SPDX-License-Identifier: BSD-3-Clause-Clear
'''

from math import sqrt
import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import HeteroConv, Linear, LayerNorm, TransformerConv
from utils import get_sinr


# %% Data Module
class GNNDataModule(pl.LightningDataModule):

    def __init__(self, dataset_name, train_batch_size, val_batch_size,
                 test_batch_size):
        super().__init__()
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.dataset_name = dataset_name
        self.num_workers = 0
        _, self.files_split_dict = torch.load(self.dataset_name)
        self.filenames = list(self.files_split_dict.keys())

    def setup(self, stage=None):
        self.dataset, files_split_dict = torch.load(self.dataset_name)
        self.train_set = []
        self.val_set = []
        self.test_set = []

        for file_name in files_split_dict.keys():
            tmp_data = self.dataset[file_name]
            split = files_split_dict[file_name]
            if stage == 'fit':
                self.train_set.extend(tmp_data[:int(split[0]*len(tmp_data))])
                self.val_set.append(
                    tmp_data[int(split[0]*len(tmp_data)):
                             int((split[0]+split[1])*len(tmp_data))])
            if stage == 'test':
                self.test_set.append(
                    tmp_data[int((split[0]+split[1])*len(tmp_data)):
                             int((split[0]+split[1]+split[2])
                                 * len(tmp_data))])

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.train_batch_size,
                          shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return [DataLoader(elem, batch_size=self.val_batch_size,
                           shuffle=False, num_workers=self.num_workers)
                for elem in self.val_set]

    def test_dataloader(self):
        return [DataLoader(elem, batch_size=self.test_batch_size,
                           shuffle=False, num_workers=self.num_workers)
                for elem in self.test_set]


# %% GNN Modules
class CoreGNNHeteroModule(pl.LightningModule):
    def __init__(self, train_batch_size, val_batch_size,
                 test_batch_size, files_dict, lr, hc, heads, **kwargs):
        super().__init__()
        self.save_hyperparameters("train_batch_size", "val_batch_size",
                                  "test_batch_size", "files_dict",
                                  "lr", "hc", "heads")
        self.save_hyperparameters(kwargs)

        self.filenames = []
        for filename in files_dict.keys():
            self.filenames.append(filename)
        self.batch_size = train_batch_size
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.lr = lr
        self.heads = heads
        self.hc = hc

        self.val_step_outputs = [[] for x in range(len(self.filenames))]

    def common_step(self, batch):
        y_hat = self(batch)
        y = batch['channel'].y
        return y_hat, y

    def training_step(self, batch, batch_idx):

        y_hat, y = self.common_step(batch)
        SINR, SINR_hat = get_sinr(batch, y_hat)

        train_loss = F.mse_loss(SINR_hat, SINR, reduction='mean')
        acc = torch.abs((SINR-SINR_hat)/SINR)
        self.log('acc', 1-acc.mean(),
                 batch_size=self.train_batch_size, prog_bar=True)
        self.log("train_loss", train_loss, batch_size=self.train_batch_size)
        return train_loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):

        y_hat, y = self.common_step(batch)
        SINR, SINR_hat = get_sinr(batch, y_hat)

        idx = torch.cumsum(batch['channel'].n_ues, dim=0)
        device = SINR_hat.device
        SINR_hat = torch.tensor(
            [min(SINR_hat[idx[i]-batch['channel'].n_ues[i]:idx[i]]).item()
             for i in range(len(idx))]).to(device)
        acc = torch.abs((batch['channel'].sinr-SINR_hat)/batch['channel'].sinr)
        acc = 1-acc.mean()
        self.log('val_acc_{}'.format(self.filenames[dataloader_idx]),
                 acc, add_dataloader_idx=False,
                 batch_size=self.val_batch_size, prog_bar=False)
        val_loss = F.mse_loss(
            SINR_hat, batch['channel'].sinr, reduction='mean')
        self.log('val_loss_{}'.format(self.filenames[dataloader_idx]),
                 val_loss, add_dataloader_idx=False,
                 batch_size=self.val_batch_size, prog_bar=False)

        # Save the validation loss on this dataset to be used in the method
        # on_validation_epoch_end()
        self.val_step_outputs[dataloader_idx].append(acc)

        return val_loss

    # Log the average loss over all validation datasets (outputs of all
    # validation_step calls)
    def on_validation_epoch_end(self):
        flat_list = []
        for idx in range(len(self.val_step_outputs)):
            flat_list.extend(self.val_step_outputs[idx])
            # Free memory
            self.val_step_outputs[idx].clear()
        avg_loss = sum(flat_list) / len(flat_list)
        self.log("hp_metric", avg_loss, prog_bar=True)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        y_hat, y = self.common_step(batch)
        SINR, SINR_hat = get_sinr(batch, y_hat)

        idx = torch.cumsum(batch['channel'].n_ues, dim=0)
        device = SINR_hat.device
        SINR_hat = torch.tensor(
            [min(SINR_hat[idx[i]-batch['channel'].n_ues[i]:idx[i]]).item()
             for i in range(len(idx))]).to(device)
        acc = torch.abs((batch['channel'].sinr-SINR_hat)/batch['channel'].sinr)
        self.log('test_acc_{}'.format(self.filenames[dataloader_idx]),
                 1-acc.mean(), add_dataloader_idx=False,
                 batch_size=self.test_batch_size, prog_bar=False)
        test_loss = F.mse_loss(
            SINR_hat, batch['channel'].sinr, reduction='mean')
        self.log('test_loss_{}'.format(self.filenames[dataloader_idx]),
                 test_loss, add_dataloader_idx=False,
                 batch_size=self.test_batch_size, prog_bar=False)
        return test_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


class GNNLinearPrecodingPyG(CoreGNNHeteroModule):
    def __init__(self, train_batch_size, val_batch_size, test_batch_size,
                 files_dict, lr, hc, heads):
        files_dict = files_dict
        super().__init__(train_batch_size, val_batch_size,
                         test_batch_size, files_dict, lr, hc, heads)
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        hc = hc
        heads = heads
        aggr = 'sum'
        num_layers = len(hc)

        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        for i in range(num_layers-1):
            in_channels = hc[i]
            out_channels = int(hc[i+1] / heads)
            conv = HeteroConv({
                ('channel', 'same_ue', 'channel'):
                TransformerConv(in_channels, out_channels,
                                heads=heads, dropout=0.0, root_weight=True),
                ('channel', 'same_ap', 'channel'):
                TransformerConv(in_channels, out_channels,
                                heads=heads, dropout=0.0, root_weight=True)
            }, aggr=aggr)
            self.convs.append(conv)
            self.norms.append(LayerNorm(hc[i+1]))

        self.lin = Linear(hc[-1], 6)

    def reset_parameters(self):
        for conv, norm in zip(self.convs, self.norms):
            conv.reset_parameters()
            norm.reset_parameters()
        self.lin.reset_parameters()

    def forward(self, batch):
        if hasattr(batch['channel'], 'batch'):
            channel_batch = batch['channel'].batch
        else:
            channel_batch = None
        x_dict = batch.x_dict
        edge_index_dict = batch.edge_index_dict
        for conv, norm in zip(self.convs, self.norms):
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {'channel': norm(x_dict['channel'].relu(), channel_batch)}
        return self.lin(x_dict['channel'])


class FastGNNLinearPrecodingLightning(CoreGNNHeteroModule):
    def __init__(self, train_batch_size, val_batch_size, test_batch_size,
                 files_dict, lr, hc, heads):
        files_dict = files_dict
        super().__init__(train_batch_size, val_batch_size,
                         test_batch_size, files_dict, lr, hc, heads)
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        heads = heads

        self.model = FastGNNLinearPrecoding(hc, heads)

    def reset_parameters(self):
        self.model.reset_parameters()

    def forward(self, batch):
        x = batch['channel'].x
        edge_index_ue = batch['channel', 'same_ue', 'channel'].edge_index
        edge_index_ap = batch['channel', 'same_ap', 'channel'].edge_index

        return self.model(x, edge_index_ue, edge_index_ap)


class FastGNNLinearPrecoding(torch.nn.Module):

    def __init__(self, hc, heads):
        '''
        Implementation of a single attention head GNNLinearPrecoding without
        PyG. This implementation can be compiled and has faster inference than
        GNNLinearPrecoding.

        Parameters
        ----------
        hc :        list of layer sizes.
        heads :     Not used. # TODO

        '''
        super().__init__()

        # TODO: heads not used at the moment (single head implementation only)
        self.heads = 1
        self.hc = hc
        self.num_layers = len(hc)-1
        self.convs1 = torch.nn.ModuleList()
        self.convs2 = torch.nn.ModuleList()
        self.convs3 = torch.nn.ModuleList()
        self.convs4 = torch.nn.ModuleList()
        self.convs5 = torch.nn.ModuleList()
        self.convs6 = torch.nn.ModuleList()
        self.convs7 = torch.nn.ModuleList()
        self.convs8 = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        self.relu = torch.nn.ReLU()
        for i in range(self.num_layers):
            self.convs1.append(torch.nn.Linear(
                hc[i], int(hc[i+1]), bias=True))
            self.convs2.append(torch.nn.Linear(
                hc[i], int(hc[i+1]), bias=True))
            self.convs3.append(torch.nn.Linear(
                hc[i], int(hc[i+1]), bias=True))
            self.convs4.append(torch.nn.Linear(
                hc[i], int(hc[i+1]), bias=True))
            self.convs5.append(torch.nn.Linear(
                hc[i], int(hc[i+1]), bias=True))
            self.convs6.append(torch.nn.Linear(
                hc[i], int(hc[i+1]), bias=True))
            self.convs7.append(torch.nn.Linear(
                hc[i], int(hc[i+1]), bias=True))
            self.convs8.append(torch.nn.Linear(
                hc[i], int(hc[i+1]), bias=True))
            self.norms.append(torch.nn.LayerNorm(hc[i+1]))

        self.lin = torch.nn.Linear(hc[-1], 6)

    def forward(self, x, edge_index_ue, edge_index_ap):
        device = x.device
        num_nodes = x.shape[0]
        tensor_zeros = torch.zeros(
            (num_nodes, ), dtype=x.dtype, device=device)

        for conv1, conv2, conv3, conv4, conv5, conv6,\
            conv7, conv8, norm in zip(self.convs1, self.convs2, self.convs3,
                                      self.convs4, self.convs5, self.convs6,
                                      self.convs7, self.convs8, self.norms):

            # edge index ue
            row_j, row_i = edge_index_ue[0][:], edge_index_ue[1][:]
            x1 = conv1(x)
            x2 = conv2(x)
            x3 = conv3(x)
            x4 = conv4(x)

            x2 = x2[row_j]
            x3 = x3[row_i]
            x4 = x4[row_j]
            d = int(x1.shape[1]/self.heads)
            zeros_repeat = torch.zeros(
                (num_nodes, d), dtype=x.dtype, device=device)
            x_3_4 = x3*x4

            alpha_num = torch.sum(x_3_4, dim=1, dtype=x_3_4.dtype)
            alpha_num = torch.exp(alpha_num/sqrt(d))

            alpha_den = torch.scatter_add(tensor_zeros, 0, row_i, alpha_num)

            alpha_den = alpha_den[row_i]
            alpha = alpha_num/alpha_den

            alpha = alpha.unsqueeze(1)
            alpha_x2 = alpha*x2

            out = torch.scatter_add(
                zeros_repeat, 0, row_i.unsqueeze(1).expand(-1, d), alpha_x2)

            OUT = out+x1

            # edge index ap
            row_j, row_i = edge_index_ap[0][:], edge_index_ap[1][:]
            x5 = conv5(x)
            x6 = conv6(x)
            x7 = conv7(x)
            x8 = conv8(x)

            x6 = x6[row_j]
            x7 = x7[row_i]
            x8 = x8[row_j]

            x_7_8 = x7*x8

            alpha_num = torch.sum(x_7_8, dim=1, dtype=x_7_8.dtype)
            alpha_num = torch.exp(alpha_num/sqrt(d))
            alpha_den = torch.scatter_add(tensor_zeros, 0, row_i, alpha_num)

            alpha_den = alpha_den[row_i]
            alpha = alpha_num/alpha_den
            alpha = alpha.unsqueeze(1)
            alpha_x6 = alpha*x6

            out = torch.scatter_add(
                zeros_repeat, 0, row_i.unsqueeze(1).expand(-1, d), alpha_x6)

            OUT = OUT+out+x5
            x = norm(self.relu(OUT))
        return self.lin(x)
