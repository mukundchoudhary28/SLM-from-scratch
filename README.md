# SLM from Scratch

A GPT-style Small Language Model implemented from scratch in PyTorch, trained on the [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset. The project is structured as a production Python package with a modular architecture, CLI entrypoint, and a full training pipeline.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red)
![Model](https://img.shields.io/badge/Model-30M%20Params-green)
![Status](https://img.shields.io/badge/Status-Active-success)

---

## Architecture

![SLM architecture](plots/architecture.svg)

| Hyperparameter | Value |
|---|---|
| Architecture | Decoder-only transformer |
| Parameters | ~30M |
| Layers | 6 |
| Attention heads | 6 |
| Embedding dim | 384 |
| Context window | 128 tokens |
| Vocabulary | 50,304 (GPT-2 BPE) |
| Dataset | TinyStories |

---

## Project structure

```
slm/
├── src/
│   ├── attention.py       # Causal multi-head self-attention
│   ├── blocks.py          # LayerNorm, MLP, Transformer block
│   ├── model.py           # Full GPT model
│   ├── config.py          # Pydantic config dataclasses
│   ├── data_pipeline.py   # Download, tokenize, write .bin files
│   ├── data_loader.py     # Memory-mapped batch sampling
│   ├── trainer.py         # Training loop, optimiser, scheduler
│   └── utils.py           # Loss plotting, number formatting
├── notebooks/
│   └── walkthrough.ipynb  # Annotated notebook — Google Colab ready
├── tests/
├── plots/
│   └── loss_curve.png
├── main.py                # CLI training entrypoint
└── generate.py            # Inference entrypoint
```

---

## Quickstart

```bash
# Install dependencies
pip install -e .

# Train with defaults
python main.py train

# Override hyperparameters via CLI
python main.py train --lr 3e-4 --n_layer 8 --n_embd 512 --max_iters 30000

# Generate text from checkpoint using
python main.py generate --prompt "Once upon a time" --temperature 0.8 --max_new_tokens 100
```

To train on GPU without a local setup, open `notebooks/walkthrough.ipynb` in Google Colab (set runtime to T4).

---

## Key design decisions

**Pre-norm residual connections** — LayerNorm is applied before each sublayer rather than after. More stable gradients, especially for deeper networks. This is the convention followed by GPT-2 and most modern LLMs.

**Flash Attention** — uses `F.scaled_dot_product_attention` (PyTorch ≥ 2.0) when available. Avoids materialising the full attention matrix, reducing peak VRAM. Falls back to manual masked attention for older PyTorch versions.

**Weight tying** — the token embedding matrix and LM head share weights. Cuts parameter count roughly in half and consistently improves perplexity at this scale.

**Fused QKV projection** — Q, K, V are computed in a single `Linear(n_embd, 3 × n_embd)` and split, rather than three separate projections. More efficient on GPU.

**Gradient accumulation** — effective batch size of 1,024 (`batch_size=32 × accumulation_steps=32`) without the VRAM cost.

**Mixed precision (AMP)** — `bfloat16` on supported GPUs, `float16` with `GradScaler` on older CUDA, `float32` on CPU.

**Warmup + cosine LR decay** — 1,000-step linear warmup then cosine decay from `1e-4` to `5e-5`.

**Memory-mapped data loading** — tokenized data stored as flat `uint16` `.bin` files and read via `numpy.memmap`. Scales to billion-token datasets without RAM overhead.

---

## Training curves

![Loss curve](plots/loss_curve.png)

---

## Tests

```bash
pytest tests/
```

---

## References

- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (2017)
- Radford et al., [GPT-2](https://openai.com/research/better-language-models) (2019)
- Eldan & Li, [TinyStories](https://arxiv.org/abs/2305.07759) (2023)
- Karpathy, [nanoGPT](https://github.com/karpathy/nanoGPT)