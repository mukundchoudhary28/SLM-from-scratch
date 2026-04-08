import torch
from src.model import GPT
from src.config import GPTConfig


def test_forward_pass():
    config = GPTConfig(block_size=8, vocab_size=100, n_layer=2, n_head=2, n_embd=32)
    model = GPT(config)

    x = torch.randint(0, 100, (4, 8))  # (batch, seq)
    logits, loss = model(x, x)
    assert logits.shape == (4, 8, 100)
    assert loss is not None


def test_generate():
    config = GPTConfig(block_size=8, vocab_size=100, n_layer=2, n_head=2, n_embd=32)
    model = GPT(config)
    x = torch.randint(0, 100, (1, 5))
    out = model.generate(x, max_new_tokens=5)
    assert out.shape[1] == 10  # 5 original + 5 generated


def test_eval_mode():
    config = GPTConfig()
    model = GPT(config)
    model.eval()
    x = torch.randint(0, config.vocab_size, (2, config.block_size))

    with torch.no_grad():
        logits, _ = model(x)

    assert logits is not None