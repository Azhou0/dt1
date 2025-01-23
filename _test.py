import torch
from model.backbone import Accent2SpeakerLoadCheckpoint

model = Accent2SpeakerLoadCheckpoint(ckpt_path_hubert="model/hubert-base/chinese-hubert-base.pt", ckpt_path_accent="log/hubert-base-unfrozen-finetune/version_0/checkpoints/epoch=7-val_acc=0.8390.ckpt")
print(model.classifier)