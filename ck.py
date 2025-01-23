import torch
from fairseq.checkpoint_utils import load_checkpoint_to_cpu

state = load_checkpoint_to_cpu('log/epoch=5-val_acc=0.8396.ckpt', {})
print(state.keys())