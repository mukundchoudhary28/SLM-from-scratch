import torch
from contextlib import nullcontext
from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingLR
from .data_loader import get_batch
from .utils import format_num


# -----------------------------
# Device + AMP setup
# -----------------------------
def setup_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_type = device.type

    ptdtype = (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float32 if device.type == "cpu"
        else torch.float16
    )

    ctx = (
        nullcontext()
        if device_type == "cpu"
        else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    )

    return device, ctx, ptdtype


# -----------------------------
# Optimizer
# -----------------------------
def build_optimizer(model, config):
    return torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        weight_decay=config.weight_decay,
        eps=1e-9,
    )


# -----------------------------
# Scheduler (Warmup + Cosine)
# -----------------------------
def build_scheduler(optimizer, config):
    scheduler_warmup = LinearLR(optimizer, total_iters=config.warmup_steps)

    scheduler_decay = CosineAnnealingLR(
        optimizer,
        T_max=config.max_iters - config.warmup_steps,
        eta_min=config.min_lr,
    )

    return SequentialLR(
        optimizer,
        schedulers=[scheduler_warmup, scheduler_decay],
        milestones=[config.warmup_steps],
    )


# -----------------------------
# GradScaler (for fp16)
# -----------------------------
def build_scaler(device, ptdtype):
    return torch.amp.GradScaler(
        enabled=(device.type == "cuda" and ptdtype == torch.float16)
    )


# -----------------------------
# Estimate loss
# -----------------------------
def estimate_loss(model, model_config, train_config, ctx, device):
    out = {}
    model.eval()

    with torch.inference_mode():
        for split in ["train", "val"]:
            losses = torch.zeros(train_config.eval_iters)

            for k in range(train_config.eval_iters):
                X, Y = get_batch(split, model_config, train_config, device )
                X, Y = X.to(device), Y.to(device)

                with ctx:
                    _, loss = model(X, Y)

                losses[k] = loss.item()

            out[split] = losses.mean().item()

    model.train()
    return out


# -----------------------------
# Training loop
# -----------------------------
def train(model, model_config, train_config):
    device, ctx, ptdtype = setup_device()

    torch.manual_seed(train_config.seed)
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nModel Parameters:")
    print(f"Total: {format_num(total_params)}")
    print(f"Trainable: {format_num(trainable_params)}")


    optimizer = build_optimizer(model, train_config)
    scheduler = build_scheduler(optimizer, train_config)
    scaler = build_scaler(device, ptdtype)

    best_val_loss = float("inf")
    best_model_path = "checkpoints/best_model.pt"

    train_losses = []
    val_losses = []

    for iter in range(train_config.max_iters):

        # -------------------------
        # Evaluation
        # -------------------------
        if iter % train_config.eval_interval == 0 and iter > 0:
            losses = estimate_loss(model, model_config, train_config, ctx, device)

            print(
                f"Step {iter}: "
                f"train {losses['train']:.4f}, "
                f"val {losses['val']:.4f}, "
                f"lr {optimizer.param_groups[0]['lr']:.6f}"
            )

            train_losses.append(losses["train"])
            val_losses.append(losses["val"])

            # Save best model
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                torch.save({
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "iter": iter,
                    "val_loss": best_val_loss,
                }, best_model_path)

        # -------------------------
        # Gradient accumulation
        # -------------------------
        optimizer.zero_grad(set_to_none=True)

        for micro_step in range(train_config.gradient_accumulation_steps):
            X, Y = get_batch("train", model_config, train_config, device)            
            X, Y = X.to(device), Y.to(device)

            with ctx:
                _, loss = model(X, Y)
                loss = loss / train_config.gradient_accumulation_steps

            scaler.scale(loss).backward()

        # -------------------------
        # Gradient clipping (AMP safe)
        # -------------------------
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

        # -------------------------
        # Optimizer step
        # -------------------------
        scaler.step(optimizer)
        scaler.update()

        # -------------------------
        # Scheduler step
        # -------------------------
        scheduler.step()

    print("Training complete ✅")

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_loss": best_val_loss,
    }