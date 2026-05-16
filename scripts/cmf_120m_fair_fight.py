from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cmf import CMFConfig, ContinuousMeaningField
from cmf.baselines import TinyGPTLM
from cmf.checkpointing import save_model_package
from cmf.experiments import count_parameters
from cmf.runtime import resolve_device


def load_gpt2_tokenizer():
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install transformers to run the GPT-2-tokenizer fair fight.") from exc
    return AutoTokenizer.from_pretrained("gpt2")


def load_training_text(args: argparse.Namespace) -> str:
    if args.text_file:
        return args.text_file.read_text(encoding="utf-8")
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install datasets or pass --text-file.") from exc
    ds = load_dataset("wikitext", "wikitext-2-v1", split="train")
    rows = [text for text in ds[: args.rows]["text"] if len(text) > 10]
    return " ".join(rows)


def run(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    tokenizer = load_gpt2_tokenizer()
    text = load_training_text(args)
    tokens = torch.tensor(tokenizer.encode(text, add_special_tokens=True), dtype=torch.long)
    if tokens.numel() <= args.seq_len + 1:
        raise ValueError("Training corpus is too small for the requested seq_len.")
    tokens = tokens.to(device)

    cmf_config = CMFConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=768,
        hidden_dim=3072,
        num_layers=6,
        max_seq_len=args.seq_len,
    )
    gpt_config = {
        "vocab_size": tokenizer.vocab_size,
        "d_model": 768,
        "nhead": 12,
        "num_layers": 12,
        "hidden_dim": 3072,
        "dropout": 0.0,
        "max_seq_len": args.seq_len,
    }
    cmf = ContinuousMeaningField(cmf_config).to(device)
    gpt = TinyGPTLM(**gpt_config).to(device)
    cmf_opt = torch.optim.AdamW(cmf.parameters(), lr=args.lr)
    gpt_opt = torch.optim.AdamW(gpt.parameters(), lr=args.lr)

    print(f"Device: {device}")
    print(f"CMF params: {count_parameters(cmf):,}")
    print(f"GPT params: {count_parameters(gpt):,}")
    print(f"Tokens: {tokens.numel():,}")

    start = time.perf_counter()
    for step in range(args.steps):
        idx = torch.randint(0, tokens.numel() - args.seq_len - 1, (args.batch_size,), device=device)
        x = torch.stack([tokens[i : i + args.seq_len] for i in idx])
        y = torch.stack([tokens[i + 1 : i + args.seq_len + 1] for i in idx])

        cmf_opt.zero_grad(set_to_none=True)
        c_loss = cmf(x, labels=y)["loss"]
        c_loss.backward()
        cmf_opt.step()

        gpt_opt.zero_grad(set_to_none=True)
        g_loss = gpt(x, labels=y)["loss"]
        g_loss.backward()
        gpt_opt.step()

        if step % args.log_every == 0:
            elapsed = time.perf_counter() - start
            print(f"step={step} cmf={c_loss.item():.4f} gpt={g_loss.item():.4f} elapsed={elapsed:.1f}s")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(cmf.state_dict(), args.out_dir / "cmf_gpt2_203m_state_dict.pt")
    torch.save(gpt.state_dict(), args.out_dir / "gpt_gpt2_state_dict.pt")
    save_model_package(
        args.out_dir / "cmf_gpt2_203m.package.pt",
        cmf,
        model_type="continuous_cmf",
        config=cmf_config,
        tokenizer=tokenizer,
        tokenizer_name="gpt2",
        training={"steps": args.steps, "seq_len": args.seq_len, "batch_size": args.batch_size},
    )
    save_model_package(
        args.out_dir / "gpt_gpt2.package.pt",
        gpt,
        model_type="tiny_gpt",
        config=gpt_config,
        tokenizer=tokenizer,
        tokenizer_name="gpt2",
        training={"steps": args.steps, "seq_len": args.seq_len, "batch_size": args.batch_size},
    )
    print(f"Saved packages to {args.out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Strict CMF/GPT checkpoint training comparison.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--rows", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--text-file", type=Path)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "records" / "checkpoints")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
