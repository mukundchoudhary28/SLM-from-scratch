import torch
import numpy as np
import os
from .data_pipeline import run_pipeline

# -----------------------------
# Return 1 batch of data
# -----------------------------

def get_batch(split, model_config, train_config, device):

    filename = "train.bin" if split == "train" else "validation.bin"
    filepath = os.path.join("data", filename)

    if not os.path.exists(filepath):
        run_pipeline(subset=True, num_proc=1)

    data = np.memmap(filepath, dtype=np.uint16, mode="r")

    ix = torch.randint(len(data) - model_config.block_size, (train_config.batch_size,))

    x = torch.stack([
        torch.from_numpy(data[i:i + model_config.block_size].astype(np.int64))
        for i in ix
    ])

    y = torch.stack([
        torch.from_numpy(data[i + 1:i + 1 + model_config.block_size].astype(np.int64))
        for i in ix
    ])

    if device == "cuda":
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
        y = y.to(device)

    return x, y