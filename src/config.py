from pydantic import BaseModel
import os

# -----------------------------
# Model Config (Architecture)
# -----------------------------
class GPTConfig(BaseModel):
    block_size: int = 128       # context length
    vocab_size: int = 50304     # tokenizer vocab size
    n_layer: int = 6            # number of transformer blocks
    n_head: int = 6             # attention heads
    n_embd: int = 384           # embedding dimension
    dropout: float = 0.1
    bias: bool = True


    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(self.model_dump_json(indent=4))

    @classmethod
    def load(cls, path: str):
        with open(path, "r") as f:
            return cls.model_validate_json(f.read())


# -----------------------------
# Training Config
# -----------------------------
class TrainConfig(BaseModel):
    # Optimization
    learning_rate: float = 1e-4
    min_lr: float = 5e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95

    # Training schedule
    max_iters: int = 20000
    warmup_steps: int = 1000
    eval_iters: int = 500
    eval_interval: int = 500

    # Batch + sequence
    batch_size: int = 32
    gradient_accumulation_steps: int = 32

    # System
    seed: int = 42