import os
import numpy as np
from tqdm.auto import tqdm
import tiktoken
from datasets import load_dataset


# -----------------------------
# Config
# -----------------------------
DATA_DIR = "data"
TOKENIZER_NAME = "gpt2"

# -----------------------------
# Load dataset
# -----------------------------
def load_data(subset: bool = False):
    """
    Load TinyStories dataset.
    
    Args:
        subset (bool): If True, load small subset for debugging
    
    Returns:
        DatasetDict
    """
    if subset:
        return load_dataset(
            "roneneldan/TinyStories",
            split={
                "train": "train[:100]",
                "validation": "validation[:20]"
            }
        )
    else:
        return load_dataset("roneneldan/TinyStories")


# -----------------------------
# Tokenization
# -----------------------------
def tokenize_dataset(ds, num_proc: int = 1):
    """
    Tokenize dataset using tiktoken.
    """

    enc = tiktoken.get_encoding(TOKENIZER_NAME)

    def process(example):
        ids = enc.encode_ordinary(example["text"])
        return {"ids": ids, "len": len(ids)}

    tokenized = ds.map(
        process,
        remove_columns=["text"],
        desc="Tokenizing dataset",
        num_proc=num_proc,
    )

    return tokenized


# -----------------------------
# Write to .bin using memmap
# -----------------------------
def write_bin_file(dset, split: str, data_dir: str = DATA_DIR):
    """
    Write dataset split to binary file.
    """

    os.makedirs(data_dir, exist_ok=True)
    filename = os.path.join(data_dir, f"{split}.bin")

    # total number of tokens
    arr_len = np.sum(dset["len"], dtype=np.uint64)

    dtype = np.uint16
    arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))

    # avoid too many empty shards
    total_batches = min(1024, len(dset))
    idx = 0

    for batch_idx in tqdm(range(total_batches), desc=f"Writing {filename}"):

        batch = (
            dset.shard(
                num_shards=total_batches,
                index=batch_idx,
                contiguous=True
            )
            .with_format("numpy")
        )

        arr_batch = np.concatenate(batch["ids"])
        arr[idx: idx + len(arr_batch)] = arr_batch
        idx += len(arr_batch)

    arr.flush()


# -----------------------------
# Main pipeline
# -----------------------------
def run_pipeline(subset: bool = True, num_proc: int = 1):
    """
    Full pipeline:
    load → tokenize → write .bin
    """

    print("Loading dataset...")
    ds = load_data(subset=subset)

    print("Tokenizing dataset...")
    tokenized = tokenize_dataset(ds, num_proc=num_proc)

    print("Writing binary files...")
    for split, dset in tokenized.items():
        write_bin_file(dset, split)

    print("Done ✅")
