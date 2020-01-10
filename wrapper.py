import pytorch_lightning as pl
import torch.nn as nn
import torch

from models import Siamese
from RAdam import RAdam

import wandb

from torch.optim.lr_scheduler import LambdaLR


def get_linear_warmup_scheduler(optimizer, num_warmup_steps, last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1.0, num_warmup_steps))
        return 1.0

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


class LightningWrapper(pl.LightningModule):

    def __init__(self, data, config):
        super(LightningWrapper, self).__init__()
        self.config = config
        self.siamese = Siamese(model_name=config.model_name)
        self.loss = nn.CrossEntropyLoss(reduction='mean')
        self.current_val_loss = 0.
        self.train_loader, self.dev_loader, self.test_loader = data

    def forward(self, batch):
        premise, hypothesis, label = batch
        return self.siamese(premise, hypothesis)

    def training_step(self, batch, batch_nb):
        _, _, label = batch
        out = self.forward(batch)
        return {'loss': self.loss(out, label)}

    def validation_step(self, batch, batch_nb):
        _, _, label = batch
        out = self.forward(batch)
        winners = out.argmax(dim=-1)
        correct = (winners == label)
        accuracy = correct.sum().float() / float(correct.size(0))
        return {'val_loss': self.loss(out, label),
                'val_accuracy': accuracy}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_accuracy'] for x in outputs]).mean()
        print(f'Avg val loss: {avg_loss}, Avg val accuracy: {avg_acc}')
        res = {'avg_val_loss': avg_loss,
               'avg_val_accuracy': avg_acc}
        wandb.log(res)

        self.current_val_loss = avg_loss  # save current val loss state for ReduceLROnPlateau scheduler

        return res

    def configure_optimizers(self):
        self.opt = RAdam(self.siamese.parameters(),
                         lr=self.config.lr,
                         betas=self.config.betas,
                         eps=self.config.eps,
                         weight_decay=self.config.weight_decay,
                         degenerated_to_sgd=True)

        self.linear_warmup = get_linear_warmup_scheduler(self.opt,
                                                         num_warmup_steps=self.config.warmup_steps)
        self.reduce_lr_on_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.opt,
            mode='min',
            factor=0.1,
            patience=5,
            verbose=True,
            cooldown=5,
            min_lr=1e-8,
        )

        return [self.opt], [self.linear_warmup, self.reduce_lr_on_plateau]

    def optimizer_step(self, epoch_nb, batch_nb, optimizer, optimizer_i, second_order_closure=None):
        self.opt.step()
        self.opt.zero_grad()
        self.linear_warmup.step()
        if self.trainer.global_step % self.config.val_check_interval == 0:
            self.reduce_lr_on_plateau.step(self.current_val_loss)

    @pl.data_loader
    def train_dataloader(self):
        return self.train_loader

    @pl.data_loader
    def val_dataloader(self):
        return self.dev_loader

    @pl.data_loader
    def test_dataloader(self):
        return self.test_loader
