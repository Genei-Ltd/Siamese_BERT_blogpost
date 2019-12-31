import pytorch_lightning as pl
from wrapper import LightningWrapper
import wandb

from torchtext import data
from torchtext import datasets

from transformers import AutoTokenizer

if __name__ == '__main__':
    wandb.init(project='Siamese_SNLI')

    # CONFIG
    ########################################
    config = wandb.config

    # Model hyperparams
    config.model_name = 'albert-base-v1'
    config.aggr = 'mean'

    # Training hyperparams
    config.batch_size = 32
    config.epochs = 10

    # Validation hyperparams
    config.val_check_interval = 250
    config.val_percent_check = 0.3

    # Optimization Hyperparams
    config.optimizer = 'RAdam'
    config.lr = 1e-4
    config.betas = (0.9, 0.999)
    config.eps = 1e-07
    config.weight_decay = 1e-7
    config.gradient_clip_val = 15
    config.warmup_steps = 100

    config.device = 'cpu'
    config.no_cuda = True
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

    LABEL.build_vocab(train)
    num_labels = len(LABEL.vocab)

    train_iter, dev_iter, test_iter = data.BucketIterator.splits(
        (train, dev, test), batch_size=config.batch_size, device=config.device)
    ########################################

    # MODEL FITTING
    ########################################
    model = LightningWrapper(config=config,  # learning rate etc
                             data=(train_iter, dev_iter, test_iter),
                             num_labels=num_labels,
                             )

    trainer = pl.Trainer(logger=False,
                         checkpoint_callback=True,
                         early_stop_callback=True,
                         default_save_path='.',
                         gradient_clip_val=config.gradient_clip_val,
                         # gpus=1,
                         show_progress_bar=False,
                         val_check_interval=config.val_check_interval,
                         val_percent_check=config.val_percent_check,
                         max_nb_epochs=20,
                         min_nb_epochs=1
                         )

    print('Starting Model')
    trainer.fit(model)
########################################
