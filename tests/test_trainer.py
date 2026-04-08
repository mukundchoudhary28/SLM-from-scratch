from src.model import GPT
from src.config import GPTConfig, TrainConfig
from src.trainer import train


def test_training_runs():
    model_config = GPTConfig(
        block_size=8,
        vocab_size=50304,
        n_layer=2,
        n_head=2,
        n_embd=32
    )

    train_config = TrainConfig(
        max_iters=2,
        eval_interval=1,
        batch_size=2
    )

    model = GPT(model_config)

    results = train(model, model_config, train_config)

    assert "train_losses" in results
    assert len(results["train_losses"]) > 0