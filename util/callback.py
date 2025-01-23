import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
from lightning.pytorch.callbacks import BasePredictionWriter
from util.unique_labels import unique_labels


class OverrideEpochStepCallback(pl.callbacks.Callback):
    """
    Override the step axis in Tensorboard with epoch. Just ignore the warning message popped out.
    """
    def __init__(self) -> None:
        super().__init__()

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self._log_step_as_current_epoch(trainer, pl_module)

    def on_test_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self._log_step_as_current_epoch(trainer, pl_module)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self._log_step_as_current_epoch(trainer, pl_module)

    def _log_step_as_current_epoch(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        pl_module.log("step", trainer.current_epoch)


class FreezeEncoderFinetuneClassifier(pl.callbacks.Callback):
    """
    Freeze the encoder of a model while fine-tuning the classifier.
    """
    def __init__(self) -> None:
        super().__init__()

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        for layer in pl_module.backbone.classifier.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        # Freeze all parameters
        for param in pl_module.backbone.parameters():
            param.requires_grad = False
        pl_module.backbone.eval()
        # Unfreeze the parameters of the classifier
        for param in pl_module.backbone.classifier.parameters():
            param.requires_grad = True
        pl_module.backbone.classifier.train()


def read_pair_wav_txt(file_path):
    pair_wav = []
    with open(file_path, 'r') as f:
        for line in f:
            data = line.strip().split()  # 使用空白字符分割每一行
            wav1, wav2, label = data[0], data[1], data[2]
            pair_wav.append([wav1, wav2, label])
    return pair_wav


class PredictionWriter(BasePredictionWriter):
    """
    Write the predictions of a pretrained model into a pt file.
    """
    def __init__(self, output_dir, meta_dir, predict_subset, write_interval="epoch"):
        super().__init__(write_interval)
        self.output_dir = output_dir
        self.meta_dir = meta_dir
        self.predict_subset = predict_subset

    def write_on_batch_end(
        self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx
    ):
        pass

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        predictions = torch.cat(predictions, dim=0).cpu().numpy().tolist()
        # Get filenames
        meta = read_pair_wav_txt(f"{self.meta_dir}/{self.predict_subset}.txt")
        for i in range(len(meta)):
            meta[i].append(predictions[i])
        with open(f'{self.output_dir}/predictions.{self.predict_subset}', 'w') as f:
            for m in meta:
                f.write(f"{m[0]} {m[1]} {m[2]} {m[3]}\n")


class AccentComparisonWriter(BasePredictionWriter):
    def __init__(self, output_dir, meta_dir, predict_subset, write_interval="epoch", similarity_threshold=0.9):
        super().__init__(write_interval)
        self.output_dir = output_dir
        self.meta_dir = meta_dir
        self.predict_subset = predict_subset
        self.similarity_threshold = similarity_threshold

    def write_on_batch_end(
        self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx
    ):
        pass

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        # Get and save the predictions
        predictions = pl_module.pred_step_outputs
        y_hat1 = np.array(predictions['y_hat1'])
        y_hat2 = np.array(predictions['y_hat2'])
        sim = np.array(predictions['sim'])
        # Get filenames
        meta = read_pair_wav_txt(f"{self.meta_dir}/{self.predict_subset}.txt")
        filenames_wav1 = [file1 for file1, _ in meta]
        filenames_wav2 = [file2 for _, file2 in meta]
        # Get the predicted index of accent labels
        _, predicted_indices1 = torch.max(torch.from_numpy(y_hat1), 1)
        _, predicted_indices2 = torch.max(torch.from_numpy(y_hat2), 1)
        # Transfer index to accent label
        accent_labels = unique_labels['subdialect']
        predicted_labels1 = []
        for i in predicted_indices1:
            predicted_labels1.append(accent_labels[i])
        predicted_labels2 = []
        for i in predicted_indices2:
            predicted_labels2.append(accent_labels[i])
        # Convert size from (N,) to (N, 1)
        predicted_labels1_arr = np.array(predicted_labels1).reshape(-1, 1)
        predicted_labels2_arr = np.array(predicted_labels2).reshape(-1, 1)
        # When two predicted labels are different and the accent similarity is less than a threshold,
        # corresponding speeches are identified as different accents
        contrast_difference = (predicted_labels1_arr != predicted_labels2_arr) & (sim < self.similarity_threshold)
        # True denotes same accent, False denotes different accents
        accent_contrast_result = ~contrast_difference
        # Make a table
        c1c2 = np.stack((filenames_wav1, predicted_labels1), axis=1)
        c3c4 = np.stack((filenames_wav2, predicted_labels2), axis=1)
        y_hat1 = np.round(y_hat1, 4)
        y_hat2 = np.round(y_hat2, 4)
        table = np.concatenate((c1c2, y_hat1, c3c4, y_hat2, accent_contrast_result, sim), axis=1)
        columns = ['filename1', 'accent_label1']
        columns.extend([l+"1" for l in accent_labels])
        columns.extend(['filename2', 'accent_label2'])
        columns.extend([l+"2" for l in accent_labels])
        columns.extend(['accent_contrast', 'similarity'])
        pd_data = pd.DataFrame(table, columns=columns)
        pd_data.to_csv(self.output_dir + f'/{self.predict_subset}_output.csv', index=False, sep='\t')
        print('\nSuccessfully save result!')