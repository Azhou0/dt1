# main.py
from lightning.pytorch.cli import LightningCLI
import torch
import data
import model.backbone
import model.lit_model
import util

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

cli = LightningCLI()