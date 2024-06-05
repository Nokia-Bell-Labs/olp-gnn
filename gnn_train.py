'''
© 2024 Nokia
Licensed under the BSD 3-Clause Clear License
SPDX-License-Identifier: BSD-3-Clause-Clear
'''

import torch
from gnn import FastGNNLinearPrecodingLightning, GNNLinearPrecodingPyG
from gnn import GNNDataModule
import pytorch_lightning as pl


torch.multiprocessing.set_sharing_strategy('file_system')

pl.seed_everything(42, workers=True)

train_batch_size = 16
val_batch_size = 16
test_batch_size = 1
lr = 7e-4
hc = [4, 16, 16, 32, 32, 16, 16]
heads = 1
dataset = 'dataset_train.pt'
dm = GNNDataModule(dataset, train_batch_size, val_batch_size, test_batch_size)


model = FastGNNLinearPrecodingLightning(train_batch_size, val_batch_size,
                                        test_batch_size, dm.files_split_dict,
                                        lr, hc, heads)

checkpoint_callback = \
    pl.callbacks.ModelCheckpoint(monitor="train_loss", save_last=True,
                                 every_n_train_steps=100, save_top_k=1,
                                 save_on_train_epoch_end=True)

trainer = pl.Trainer(max_epochs=1000, accelerator='gpu', devices=1,
                     callbacks=[checkpoint_callback])
trainer.fit(model=model, datamodule=dm)
