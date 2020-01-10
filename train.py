from utils import collate_function

import pytorch_lightning as pl
from wrapper import LightningWrapper

import wandb

from torch.utils.data import DataLoader
from torchtext import data
from torchtext import datasets

from transformers import AutoTokenizer

import os
os.environ['WANDB_API_KEY']='6beb9ef2d63f9b90456e658843c4e65ee59b88a9'

import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Siamese BERT')
    parser.add_argument('--model-name', type=str, default='bert-base-uncased')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--bs', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--distributed-backend', type=str, default=None)
    args = parser.parse_args()

    wandb.init(project='Siamese_SNLI')

    # CONFIG
    ########################################
    config = wandb.config

    # Model hyperparams
    config.model_name = args.model_name # default is bert-base-uncased
    config.aggr = 'mean'

    # Training hyperparams
    config.batch_size = args.bs # default 32
    config.epochs = args.epochs # default  5

    # Validation hyperparams
    config.val_check_interval = 250
    config.val_percent_check = 0.3

    # Optimization Hyperparams
    config.optimizer = 'RAdam'
    config.lr = args.lr
    config.betas = (0.9, 0.999)
    config.eps = 1e-07
    config.weight_decay = 1e-7
    config.gradient_clip_val = 15
    config.warmup_steps = 100

    # GPU Params
    config.gpus = args.gpus # default 1
    config.distributed_backend = args.distributed_backend
    config.no_cuda = False

    config = argparse.Namespace(**dict(config))
    ########################################

    # DATA LOADING
    ########################################
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, do_lower_case=True)

    def preprocessor(batch):
        return tokenizer.encode(batch, add_special_tokens=True)

    TEXT = data.Field(
        use_vocab=False,
        batch_first=True,
        pad_token=tokenizer.pad_token_id,
        preprocessing=preprocessor
    )
    LABEL = data.Field(sequential=False)

    train, dev, test = datasets.SNLI.splits(TEXT, LABEL)

    train_loader = DataLoader(train, batch_size=config.batch_size, collate_fn=collate_function, pin_memory=True)
    dev_loader = DataLoader(dev, batch_size=config.batch_size, collate_fn=collate_function, pin_memory=True)
    test_loader = DataLoader(test, batch_size=config.batch_size, collate_fn=collate_function, pin_memory=True)

    ########################################

    # MODEL FITTING
    ########################################
    model = LightningWrapper(config=config,  # learning rate etc
                             data=(train_loader, dev_loader, test_loader) # data
                             )

    trainer = pl.Trainer(logger=False,
                         checkpoint_callback=True,
                         early_stop_callback=True,
                         default_save_path='.',
                         gradient_clip_val=config.gradient_clip_val,
                         gpus=config.gpus,
                         distributed_backend=config.distributed_backend,
                         show_progress_bar=False,
                         val_check_interval=config.val_check_interval,
                         val_percent_check=config.val_percent_check,
                         max_nb_epochs=20,
                         min_nb_epochs=1
                         )

    print('Starting Model')
    trainer.fit(model)
########################################
