import torch
from typing import Tuple
import numpy as np

# load data
with open('train.bin', 'rb') as f: 
    train_data = np.fromfile(f, dtype=np.int16)

with open('val.bin', 'rb') as f:
    val_data = np.fromfile(f, dtype=np.int16)

# hyperparameters
block_size = 8 # context length
batch_size = 4 # n independent sequences processed in parallel

def get_batch(split: str) -> Tuple[torch.Tensor, torch.Tensor]:
    # generate a small batch of data of inputs x and targets y
    data = torch.Tensor(train_data) if split == 'train' else torch.Tensor(val_data)
    ix = torch.randint(len(data)-block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x, y # size([batch_size, block_size])

print(get_batch('train')[0].size())


