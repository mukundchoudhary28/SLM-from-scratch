import torch
import tiktoken
from src.config import GPTConfig
from src.model import GPT


def load_model(checkpoint_path, device):
    model_config = GPTConfig.load("checkpoints/config.json")
    model = GPT(model_config)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])

    model.to(device)
    model.eval()
    return model


def generate_text(model, prompt, max_new_tokens=100, temperature=0.8,device="cpu"):
    enc = tiktoken.get_encoding("gpt2")

    # Encode input
    input_ids = enc.encode(prompt)
    input_ids = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)

    # Generate
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=50
        )

    # Decode output
    output_text = enc.decode(output_ids[0].tolist())
    return output_text


def inference(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model("checkpoints/best_model.pt", device)
    output = generate_text(model, args.prompt, max_new_tokens=args.max_new_tokens, temperature=args.temperature, device=device)
    print("\n--- Generated Text ---\n")
    print(output)
