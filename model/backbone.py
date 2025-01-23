import torch
from torch import nn
import torch.nn.functional as F
from fairseq import checkpoint_utils


class _BaseBackbone(nn.Module):
    """ Base Module for backbones. """


class SingleLinearClassifier(nn.Module):
    def __init__(self, in_features: int, num_classes: int):
        super(SingleLinearClassifier, self).__init__()
        self.linear = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.linear(x)
        x = x.mean(dim=1)
        return x


class MultiLayerPerception(nn.Module):
    def __init__(self, in_features: int, hidden_units: int, num_classes: int, dropout: float):
        super(MultiLayerPerception, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_units)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_units, num_classes)

    def forward(self, x):
        x = self.act(x)
        x = self.drop(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = x.mean(dim=1)
        return x


def postprocess(feats, normalize=False):
    if normalize:
        with torch.no_grad():
            feats = F.layer_norm(feats, feats.shape)
    return feats


class HubertBaseLoadCheckpoint(_BaseBackbone):
    def __init__(self, ckpt_path: str, num_classes: int):
        super().__init__()
        print("Loading model(s) from {}".format(ckpt_path))
        models, self.saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task([ckpt_path], suffix="",)
        print("Loaded model(s) from {}".format(ckpt_path))
        print(f"Normalize: {self.saved_cfg.task.normalize}")
        self.encoder = models[0]
        # Create a new linear classifier
        self.classifier = SingleLinearClassifier(in_features=self.saved_cfg.model.encoder_embed_dim, num_classes=num_classes)

    def forward(self, x):
        x = postprocess(x, normalize=self.saved_cfg.task.normalize)
        padding_mask = (
            torch.BoolTensor(x.shape).fill_(False)
        )
        inputs = {
            "source": x,
            "padding_mask": padding_mask.to(x.device),
        }
        feats = self.encoder.extract_features(**inputs)[0]
        return self.classifier(feats), feats


class Accent2SpeakerLoadCheckpoint(_BaseBackbone):
    def __init__(self, ckpt_path_hubert: str, ckpt_path_accent: str, num_classes: int):
        super().__init__()
        self.num_classes = num_classes

        print("Loading model(s) from {}".format(ckpt_path_accent))
        backbone = HubertBaseLoadCheckpoint(ckpt_path=ckpt_path_hubert, num_classes=9)
        data_augmentation = {"mix_up": None}
        from model.lit_model import LitAccentIdentificationSystem
        model = LitAccentIdentificationSystem.load_from_checkpoint(ckpt_path_accent, backbone=backbone,
                                                                   data_augmentation=data_augmentation)
        self.encoder = model.backbone.encoder
        self.saved_cfg = model.backbone.saved_cfg
        # Create a new linear classifier
        embed_dim = self.saved_cfg.model.encoder_embed_dim
        self.classifier = SingleLinearClassifier(in_features=embed_dim, num_classes=self.num_classes)

    def forward(self, x):
        x = postprocess(x, normalize=self.saved_cfg.task.normalize)
        padding_mask = (
            torch.BoolTensor(x.shape).fill_(False)
        )
        inputs = {
            "source": x,
            "padding_mask": padding_mask.to(x.device),
        }
        feats = self.encoder.extract_features(**inputs)[0]
        return self.classifier(feats), feats