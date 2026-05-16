from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cmf.benchmarks import (
    CHAIN_FACTS,
    FACTS,
    TaskPrompt,
    extract_answer,
    make_mixed_reasoning_corpus,
    make_task_prompts,
)
from cmf.checkpointing import save_model_package
from cmf.config import CMFConfig
from cmf.experiments import (
    benchmark_forward,
    count_parameters,
    environment_report,
    set_seed,
    write_json,
    write_markdown_report,
)
from cmf.generation import decode_tokens, encode_to_tensor, generate_ids, trim_assistant_response
from cmf.model import DeliberativeContinuousMeaningField, ParallelContinuousMeaningField
from cmf.power import PowerMonitor
from cmf.presets import estimate_cmf_parameters, get_preset
from cmf.runtime import empty_cache, peak_memory_mb, reset_peak_memory, resolve_device, synchronize_device


RECORDS = ROOT / "records" / "evals_120m"


def load_gpt2_tokenizer():
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install transformers to run the 120M GPT-2-tokenizer gate.") from exc
    return AutoTokenizer.from_pretrained("gpt2")


def build_training_text(extra_text_file: Path | None) -> str:
    docs = []
    for path in [
        ROOT / "README.md",
        ROOT / "docs" / "architecture.md",
        ROOT / "docs" / "cmf_infinity_architecture.md",
        ROOT / "docs" / "current_claims.md",
    ]:
        if path.exists():
            docs.append(path.read_text(encoding="utf-8"))
    if extra_text_file:
        docs.append(extra_text_file.read_text(encoding="utf-8"))

    task_text = make_mixed_reasoning_corpus(seed=2026)
    instruction_text = (
        "Instruction: Answer with the exact short answer after A:. "
        "If evidence is insufficient, say unknown.\n"
        "Q: what is cuda? A: gpu\n"
        "Q: what is cmf? A: field\n"
        "Q: 7+8=? A: 15\n"
        "Fact: alice is bob. Fact: bob is developer. Q: what is alice? A: developer\n"
    )
    return "\n\n".join([task_text, instruction_text * 80, *docs])


def make_batches(
    tokens: torch.Tensor,
    *,
    seq_len: int,
    batch_size: int,
    num_batches: int,
    stride: int,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    if tokens.ndim != 1:
        raise ValueError("tokens must be 1D")
    if tokens.numel() <= seq_len + 1:
        raise ValueError("token corpus is too small")
    max_start = tokens.numel() - seq_len - 1
    cursor = 0
    batches = []
    for _ in range(num_batches):
        xs = []
        ys = []
        for _batch_idx in range(batch_size):
            start = cursor % max_start
            chunk = tokens[start : start + seq_len + 1]
            xs.append(chunk[:-1])
            ys.append(chunk[1:])
            cursor += stride
        batches.append((torch.stack(xs), torch.stack(ys)))
    return batches


@torch.no_grad()
def evaluate_loss(model: torch.nn.Module, batches, device: torch.device) -> float:
    model.eval()
    losses = []
    for x, y in batches:
        output = model(x.to(device), labels=y.to(device))
        losses.append(float(output["loss"].detach().cpu()))
    model.train()
    return sum(losses) / max(1, len(losses))


def candidate_pool(item: TaskPrompt) -> list[str]:
    if item.kind.startswith("addition"):
        return [str(value) for value in range(0, 130)]
    if item.kind == "fact":
        return sorted(set(FACTS.values()))
    return sorted(set(role for _, _, role in CHAIN_FACTS))


@torch.no_grad()
def score_candidate(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    candidate: str,
    device: torch.device,
    max_context_tokens: int,
) -> float:
    prompt_ids = encode_to_tensor(tokenizer, prompt)
    full_ids = encode_to_tensor(tokenizer, prompt + candidate + "\n")
    if full_ids.numel() < 2:
        return float("-inf")
    if full_ids.numel() > max_context_tokens:
        full_ids = full_ids[-max_context_tokens:]
        prompt_ids = prompt_ids[-min(prompt_ids.numel(), max_context_tokens - 1):]
    input_ids = full_ids[:-1].unsqueeze(0).to(device)
    labels = full_ids[1:].unsqueeze(0).to(device)
    output = model(input_ids)
    log_probs = torch.log_softmax(output["logits"], dim=-1)
    token_log_probs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    start = max(0, min(prompt_ids.numel() - 1, token_log_probs.size(1) - 1))
    return float(token_log_probs[0, start:].sum().detach().cpu())


@torch.no_grad()
def evaluate_tasks(
    model: torch.nn.Module,
    tokenizer: Any,
    prompts: list[TaskPrompt],
    device: torch.device,
    *,
    max_context_tokens: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    model.eval()
    prompt_rows = []
    candidate_rows = []
    prompt_correct = 0
    candidate_correct = 0
    by_kind: dict[str, list[bool]] = {}
    by_kind_candidate: dict[str, list[bool]] = {}

    for item in prompts:
        input_ids = encode_to_tensor(tokenizer, item.prompt).unsqueeze(0).to(device)
        output_ids = generate_ids(
            model,
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=0.3,
            top_k=20,
            top_p=0.95,
            repetition_penalty=1.05,
            eos_token_id=getattr(tokenizer, "eos_token_id", None),
            max_context_tokens=max_context_tokens,
        )
        decoded = decode_tokens(tokenizer, output_ids[0])
        answer = extract_answer(decoded)
        is_prompt_correct = answer == item.answer
        prompt_correct += int(is_prompt_correct)
        by_kind.setdefault(item.kind, []).append(is_prompt_correct)
        prompt_rows.append(
            {
                "kind": item.kind,
                "prompt": item.prompt,
                "expected": item.answer,
                "generated": decoded,
                "answer": answer,
                "correct": is_prompt_correct,
            }
        )

        scored = [
            (
                candidate,
                score_candidate(
                    model,
                    tokenizer,
                    item.prompt,
                    candidate,
                    device,
                    max_context_tokens,
                ),
            )
            for candidate in candidate_pool(item)
        ]
        scored.sort(key=lambda row: row[1], reverse=True)
        best = scored[0][0] if scored else ""
        is_candidate_correct = best == item.answer
        candidate_correct += int(is_candidate_correct)
        by_kind_candidate.setdefault(item.kind, []).append(is_candidate_correct)
        expected_rank = next(
            (idx + 1 for idx, (candidate, _score) in enumerate(scored) if candidate == item.answer),
            None,
        )
        candidate_rows.append(
            {
                "kind": item.kind,
                "prompt": item.prompt,
                "expected": item.answer,
                "best_answer": best,
                "best_score": scored[0][1] if scored else float("-inf"),
                "expected_rank": expected_rank,
                "correct": is_candidate_correct,
            }
        )

    return {
        "prompt_accuracy": prompt_correct / max(1, len(prompts)),
        "candidate_accuracy": candidate_correct / max(1, len(prompts)),
        "by_kind_prompt_accuracy": {
            kind: sum(values) / max(1, len(values))
            for kind, values in by_kind.items()
        },
        "by_kind_candidate_accuracy": {
            kind: sum(values) / max(1, len(values))
            for kind, values in by_kind_candidate.items()
        },
        "prompt_rows": prompt_rows,
        "candidate_rows": candidate_rows,
    }


@torch.no_grad()
def make_generation_samples(
    model: torch.nn.Module,
    tokenizer: Any,
    device: torch.device,
    *,
    max_context_tokens: int,
    max_new_tokens: int,
) -> list[dict[str, str]]:
    prompts = [
        "Q: what is python? A: ",
        "Q: 7+8=? A: ",
        "Fact: alice is bob. Fact: bob is developer. Q: what is alice? A: ",
        "Explain CMF in one short sentence: ",
    ]
    rows = []
    for prompt in prompts:
        input_ids = encode_to_tensor(tokenizer, prompt).unsqueeze(0).to(device)
        output_ids = generate_ids(
            model,
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            repetition_penalty=1.05,
            eos_token_id=getattr(tokenizer, "eos_token_id", None),
            max_context_tokens=max_context_tokens,
        )
        rows.append(
            {
                "prompt": prompt,
                "completion": trim_assistant_response(decode_tokens(tokenizer, output_ids[0])),
            }
        )
    return rows


def deployment_verdict(result: dict[str, Any]) -> dict[str, Any]:
    finite = math.isfinite(result.get("final_eval_loss", float("nan")))
    improved = result.get("final_eval_loss", float("inf")) < result.get("initial_eval_loss", 0.0)
    prompt_accuracy = result.get("task_eval", {}).get("prompt_accuracy", 0.0)
    candidate_accuracy = result.get("task_eval", {}).get("candidate_accuracy", 0.0)
    throughput = result.get("throughput", {}).get("tokens_per_sec", 0.0)
    package_saved = bool(result.get("package_out"))

    local_smoke_deployable = finite and package_saved and throughput > 0
    world_deployable = (
        local_smoke_deployable
        and improved
        and prompt_accuracy >= 0.50
        and candidate_accuracy >= 0.70
        and result.get("toolchain", {}).get("cmf_cuda_available", False)
    )
    return {
        "local_smoke_deployable": local_smoke_deployable,
        "world_deployable": world_deployable,
        "start_8b_training_now": False,
        "reason": (
            "120M package is usable for local smoke inference, but quality, native runtime, "
            "and infrastructure gates are not sufficient for a public world deployment."
            if local_smoke_deployable and not world_deployable
            else "The 120M smoke gate did not produce a usable package."
        ),
        "required_before_world_deploy": [
            "Train on a real corpus for many billions of tokens, not a short smoke corpus.",
            "Run contamination-aware factuality, reasoning, safety, and multilingual evals.",
            "Build and validate the C++/CUDA extension or another production inference path.",
            "Add serving, monitoring, abuse controls, rollback, and cost/latency SLOs.",
            "Run multi-seed comparisons against Transformer/Mamba baselines at matched scale.",
        ],
        "required_before_8b_training": [
            "Use multi-GPU or cloud hardware with enough VRAM for 8B optimizer states.",
            "Add distributed training/FSDP or ZeRO, checkpoint sharding, and resume tests.",
            "Prove 120M and 203M scaling on real held-out text first.",
            "Validate tokenizer/data pipeline at production token volume.",
            "Estimate training budget, eval cadence, and safety release gates.",
        ],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(args.seed)
    RECORDS.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    tokenizer = load_gpt2_tokenizer()
    preset = get_preset(args.preset)
    config = CMFConfig(**preset.config.__dict__)
    config.max_seq_len = args.seq_len
    if args.thinking_steps is not None:
        config.thinking_steps = args.thinking_steps
        config.max_thinking_steps = args.thinking_steps
        config.min_thinking_steps = min(config.min_thinking_steps, args.thinking_steps)
    if preset.model_type == "parallel_cmf":
        model = ParallelContinuousMeaningField(config).to(device)
    elif preset.model_type == "deliberative_cmf":
        model = DeliberativeContinuousMeaningField(config).to(device)
    else:
        raise ValueError(f"run_120m_gate only supports 120M parallel/deliberative presets, got {preset.model_type}")
    parameter_count = count_parameters(model)

    text = build_training_text(args.text_file)
    token_ids = encode_to_tensor(tokenizer, text)
    train_batches = make_batches(
        token_ids,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        num_batches=args.steps,
        stride=max(1, args.seq_len // 2),
    )
    eval_batches = make_batches(
        token_ids,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        num_batches=args.eval_batches,
        stride=max(1, args.seq_len * 3),
    )
    benchmark_batch = eval_batches[0]

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    empty_cache(device)
    reset_peak_memory(device)

    initial_eval = evaluate_loss(model, eval_batches, device)
    losses = []
    start = time.perf_counter()
    with PowerMonitor(interval_sec=0.1) as monitor:
        model.train()
        for step, (x, y) in enumerate(train_batches, start=1):
            optimizer.zero_grad(set_to_none=True)
            x = x.to(device)
            y = y.to(device)
            with torch.amp.autocast(device_type=device.type, enabled=args.amp and device.type == "cuda"):
                out = model(x, labels=y)
                loss = out["loss"]
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
            if step % args.log_every == 0:
                print(f"step={step} loss={losses[-1]:.4f}")
        synchronize_device(device)
    train_elapsed = time.perf_counter() - start
    train_power = monitor.summary()
    final_eval = evaluate_loss(model, eval_batches, device)

    task_prompts = make_task_prompts()[: args.prompt_limit]
    task_eval = evaluate_tasks(
        model,
        tokenizer,
        task_prompts,
        device,
        max_context_tokens=args.seq_len,
        max_new_tokens=args.max_new_tokens,
    )
    generations = make_generation_samples(
        model,
        tokenizer,
        device,
        max_context_tokens=args.seq_len,
        max_new_tokens=args.max_new_tokens,
    )

    with PowerMonitor(interval_sec=0.05) as forward_monitor:
        throughput = benchmark_forward(
            model,
            benchmark_batch[0],
            benchmark_batch[1],
            device,
            iterations=args.bench_iterations,
            warmup=1,
        )
        synchronize_device(device)
    throughput["power"] = forward_monitor.summary()

    args.package_out.parent.mkdir(parents=True, exist_ok=True)
    save_model_package(
        args.package_out,
        model,
        model_type=preset.model_type,
        config=config,
        tokenizer=tokenizer,
        tokenizer_name="gpt2",
        training={
            "scope": "120M smoke gate, not full pretraining",
            "steps": args.steps,
            "seq_len": args.seq_len,
            "batch_size": args.batch_size,
            "tokens_seen": args.steps * args.seq_len * args.batch_size,
            "initial_eval_loss": initial_eval,
            "final_eval_loss": final_eval,
        },
    )

    cmf_cuda_available = False
    try:
        import cmf_cuda  # type: ignore  # noqa: F401

        cmf_cuda_available = True
    except ImportError:
        cmf_cuda_available = False

    result: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "scope": "120M-class CMF smoke train/eval gate. This is not full pretraining.",
        "environment": environment_report(),
        "device_used": str(device),
        "preset": preset.to_dict(),
        "actual_parameters": parameter_count,
        "estimated_parameters": estimate_cmf_parameters(config, model_type=preset.model_type),
        "settings": {
            "seed": args.seed,
            "seq_len": args.seq_len,
            "batch_size": args.batch_size,
            "steps": args.steps,
            "eval_batches": args.eval_batches,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "amp": args.amp and device.type == "cuda",
            "bench_iterations": args.bench_iterations,
            "prompt_limit": args.prompt_limit,
            "max_new_tokens": args.max_new_tokens,
        },
        "corpus_tokens": int(token_ids.numel()),
        "train_losses": losses,
        "initial_eval_loss": initial_eval,
        "final_eval_loss": final_eval,
        "loss_improvement": initial_eval - final_eval,
        "perplexity": math.exp(min(final_eval, 20.0)),
        "train_elapsed_sec": train_elapsed,
        "train_tokens_per_sec": (args.steps * args.seq_len * args.batch_size) / max(train_elapsed, 1e-9),
        "train_power": train_power,
        "peak_vram_mb": peak_memory_mb(device),
        "task_eval": task_eval,
        "generation_samples": generations,
        "throughput": throughput,
        "package_out": str(args.package_out),
        "toolchain": {
            "cmf_cuda_available": cmf_cuda_available,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "cuda_vram_gb": (
                torch.cuda.get_device_properties(0).total_memory / 1024**3
                if torch.cuda.is_available()
                else None
            ),
        },
    }
    result["deployability"] = deployment_verdict(result)

    preset_slug = args.preset.replace("/", "_")
    write_json(RECORDS / "latest.json", result)
    write_markdown_report(RECORDS / "latest.md", "CMF 120M Gate", result)
    write_json(RECORDS / f"{preset_slug}_latest.json", result)
    write_markdown_report(RECORDS / f"{preset_slug}_latest.md", "CMF 120M Gate", result)
    print(json.dumps(result["deployability"], indent=2, sort_keys=True))
    print(f"Wrote {RECORDS / 'latest.json'}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/evaluate the CMF Infinity 120M-class smoke gate.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--preset", choices=["infinity-0.12b", "infinity-reasoning-0.12b"], default="infinity-0.12b")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--eval-batches", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--clip-grad-norm", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--bench-iterations", type=int, default=3)
    parser.add_argument("--prompt-limit", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--thinking-steps", type=int, help="Override deliberative preset thinking budget for smoke runs.")
    parser.add_argument("--text-file", type=Path)
    parser.add_argument(
        "--package-out",
        type=Path,
        default=ROOT / "records" / "checkpoints" / "cmf_120m_gate.package.pt",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
