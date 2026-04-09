from src.config import GPTConfig, TrainConfig
from src.model import GPT
from src.trainer import train
from src.utils import plot_losses  # if you saved plotting function separately
from generate import inference
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Train / Inference an SLM")

    subparsers = parser.add_subparsers(dest="mode", required=True)
    train_parser = subparsers.add_parser(
        "train",
        help="Train an SLM from scratch"
    )
    generate_parser = subparsers.add_parser(
        "generate",
        help="Inference using a trained SLM"
    )

# Train Parametrs

    # Training params
    train_parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    train_parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    train_parser.add_argument("--context_window", type=int, default=128, help="Block size")
    train_parser.add_argument("--max_iters", type=int, default=20000, help="Maximum training iterations")
    train_parser.add_argument("--eval_interval", type=int, default=500, help="Evaluation interval")

    # Model params
    train_parser.add_argument("--n_layer", type=int, default=6, help="Number of layers")
    train_parser.add_argument("--n_head", type=int, default=6, help="Number of attention heads")
    train_parser.add_argument("--n_embd", type=int, default=384, help="Embedding dimension")

#Generate parameters

    generate_parser.add_argument("--prompt", type=str, help="Prompt for generation", required=True)
    generate_parser.add_argument("--max_new_tokens", type=int, default=100, help="Max tokens to generate")
    generate_parser.add_argument("--temperature", type=float, default=0.8, help="Temperature for the model")

    return parser.parse_args()


def main():

    config_path = "checkpoints/config.json"
    args = parse_args()


#Inference mode
    if args.mode == "generate":
        inference(args)
        return
    
#Training mode

    # -----------------------------
    # Load configs
    # -----------------------------
    model_config = GPTConfig(
        block_size=args.context_window,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
    )
    train_config = TrainConfig(
        learning_rate=args.lr,
        batch_size=args.batch_size,
        max_iters=args.max_iters,
        eval_interval=args.eval_interval,
    )
    # -----------------------------
    # Initialize model
    # -----------------------------
    model = GPT(model_config)
    model_config.save(config_path)

    # -----------------------------
    # Train
    # -----------------------------
    results = train(model, model_config, train_config)

    # -----------------------------
    # Plot losses
    # -----------------------------
    plot_losses(
        results["train_losses"],
        results["val_losses"],
        train_config.eval_interval
    )
    

if __name__ == "__main__":
    main()