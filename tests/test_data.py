import torch
from src.data_loader import get_batch
from src.config import GPTConfig, TrainConfig


def test_get_batch_shape():
    model_config = GPTConfig(block_size=8)
    train_config = TrainConfig(batch_size=4)

    device = "cpu"

    x, y = get_batch("train", model_config, train_config, device)

    assert x.shape == (4, 8)
    assert y.shape == (4, 8)


def test_targets_are_shifted():
    model_config = GPTConfig(block_size=8)
    train_config = TrainConfig(batch_size=2)

    x, y = get_batch("train", model_config, train_config, "cpu")

    # y should be x shifted by 1
    assert torch.all(x[:, 1:] == y[:, :-1])