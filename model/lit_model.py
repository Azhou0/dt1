import lightning as L
from typing import Dict, Optional

import numpy as np
import torch
import torchinfo
import torch.nn.functional as F
from model.backbone import _BaseBackbone
from util.data_aug import _DataAugmentation
from util.lr_scheduler import exp_warmup_linear_down
from util.result_analysis import get_confusion_matrix
from lightning.pytorch.cli import OptimizerCallable


class LitAccentIdentificationSystem(L.LightningModule):
    def __init__(self,
                 backbone: _BaseBackbone,
                 data_augmentation: Dict[str, Optional[_DataAugmentation]]):
        super(LitAccentIdentificationSystem, self).__init__()
        # Save the hyperparameters for Tensorboard visualization, 'backbone' and 'spec_extractor' are excluded.
        self.save_hyperparameters(ignore=['backbone'])
        self.backbone = backbone
        self.data_aug = data_augmentation

        # Save data during testing for statistical analysis
        self._test_step_outputs = {'y': [], 'pred': []}
        # Input size of a sample, used for generating model profile.
        self._test_input_size = None

        # Save data during prediction
        self.pred_step_outputs = {'y_hat1': [], 'y_hat2': [], 'sim': []}

    @staticmethod
    def accuracy(logits, labels):
        pred = torch.argmax(logits, dim=1)
        acc = torch.sum(pred == labels).item() / len(labels)
        return acc, pred

    def forward(self, x):
        return self.backbone(x)

    def training_step(self, batch, batch_idx):
        # Load a batch of waveforms with size (N, X)
        x = batch[0]
        # Load subdialect label
        y = batch[1]
        # Instantiate data augmentation
        mix_up = self.data_aug['mix_up']
        # Apply mixup augmentation on waveform
        if mix_up is not None:
            x, y = mix_up(x, y)
        # Get the predicted labels
        y_hat, _ = self(x)
        # Calculate the loss and accuracy
        if mix_up is not None:
            pred = torch.argmax(y_hat, dim=1)
            train_loss = mix_up.lam * F.cross_entropy(y_hat, y[0]) + (1 - mix_up.lam) * F.cross_entropy(y_hat, y[1])
            corrects = (mix_up.lam * torch.sum(pred == y[0]) + (1 - mix_up.lam) * torch.sum(pred == y[1]))
            train_acc = corrects.item() / len(x)
        else:
            train_loss = F.cross_entropy(y_hat, y)
            train_acc, _ = self.accuracy(y_hat, y)
        # Log for each epoch
        self.log_dict({'train_loss': train_loss, 'train_acc': train_acc}, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return train_loss

    def validation_step(self, batch, batch_idx):
        x = batch[0]
        y = batch[1]
        y_hat, _ = self(x)
        val_loss = F.cross_entropy(y_hat, y)
        val_acc, _ = self.accuracy(y_hat, y)
        self.log_dict({'val_loss': val_loss, 'val_acc': val_acc}, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return val_acc

    def test_step(self, batch, batch_idx):
        x = batch[0]
        y = batch[1]
        # Get the input size of feature for measuring model profile
        self._test_input_size = (1, x.size(-1))
        y_hat, _ = self(x)
        test_loss = F.cross_entropy(y_hat, y)
        test_acc, pred = self.accuracy(y_hat, y)
        self.log_dict({'test_loss': test_loss, 'test_acc': test_acc})

        self._test_step_outputs['y'] += y.cpu().numpy().tolist()
        self._test_step_outputs['pred'] += pred.cpu().numpy().tolist()
        return test_acc

    def on_test_epoch_end(self):
        tensorboard = self.logger.experiment
        # Summary the model profile
        print("\n Model Profile:")
        model_profile = torchinfo.summary(self.backbone, input_size=self._test_input_size)
        macc = model_profile.total_mult_adds
        params = model_profile.total_params
        print('MACC:\t \t %.6f' % (macc / 1e6), 'M')
        print('Params:\t \t %.3f' % (params / 1e3), 'K\n')
        # Generate an confusion matrix figure
        from util.unique_labels import unique_labels
        cm = get_confusion_matrix(self._test_step_outputs, unique_labels['subdialect'])
        tensorboard.add_figure('confusion_matrix', cm)

    def predict_step(self, batch):
        x1 = batch[0]
        x2 = batch[1]
        y_hat1, emb1 = self(x1)
        y_hat2, emb2 = self(x2)
        sim = F.cosine_similarity(y_hat1, y_hat2, dim=-1)
        sim = sim.reshape(-1, 1)
        y_hat1 = F.softmax(y_hat1)
        y_hat2 = F.softmax(y_hat2)

        self.pred_step_outputs['y_hat1'] += y_hat1.cpu().numpy().tolist()
        self.pred_step_outputs['y_hat2'] += y_hat2.cpu().numpy().tolist()
        self.pred_step_outputs['sim'] += sim.cpu().numpy().tolist()
        return sim


class LitAscWithWarmupLinearDownScheduler(LitAccentIdentificationSystem):
    """
    ASC system with warmup-linear-down scheduler.
    """
    def __init__(self, optimizer: OptimizerCallable, warmup_len=4, down_len=26, min_lr=0.005, **kwargs):
        super(LitAscWithWarmupLinearDownScheduler, self).__init__(**kwargs)
        self.optimizer = optimizer
        self.warmup_len = warmup_len
        self.down_len = down_len
        self.min_lr = min_lr

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters())
        schedule_lambda = exp_warmup_linear_down(self.warmup_len, self.down_len, self.warmup_len, self.min_lr)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule_lambda)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


class LitAccent2Speaker(LitAccentIdentificationSystem):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)