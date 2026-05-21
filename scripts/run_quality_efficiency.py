from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cmf import (
    CMFConfig,
    ParallelContinuousMeaningField,
)
from cmf.baselines import TemporalConvLM, TinyGPTLM
from cmf.benchmarks import (
    make_lm_batches,
    make_mixed_reasoning_corpus,
    make_task_prompts,
    evaluate_candidate_accuracy,
    evaluate_prompt_accuracy,
    parameter_match_report,
)
from cmf.data import ByteTokenizer, fixed_eval_batches
from cmf.experiments import (
    benchmark_forward,
    count_parameters,
    environment_report,
    evaluate_loss,
    set_seed,
    train_fixed_steps,
    write_json,
    write_markdown_report,
)
from cmf.power import PowerMonitor
from cmf.runtime import empty_cache, peak_memory_mb, reset_peak_memory, resolve_device, synchronize_device


RECORDS = ROOT / "records" / "quality_efficiency"


def make_models(max_seq_len: int) -> dict[str, torch.nn.Module]:
    return {
        "parallel_cmf": ParallelContinuousMeaningField(
            CMFConfig(
                vocab_size=256,
                d_model=80,
                hidden_dim=160,
                num_layers=3,
                solver_steps_per_token=1,
                max_seq_len=max_seq_len,
                dropout=0.0,
                tie_embeddings=False,
            )
        ),
        "transformer": TinyGPTLM(
            vocab_size=256,
            d_model=88,
            nhead=4,
            num_layers=5,
            hidden_dim=176,
            dropout=0.0,
            max_seq_len=max_seq_len,
        ),
    }


def train_and_measure(
    name: str,
    model: torch.nn.Module,
    text: str,
    eval_batches,
    device: torch.device,
    seq_len: int,
    batch_size: int,
    steps: int,
    use_amp: bool,
) -> dict:
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.0)
    train_batches = make_lm_batches(
        text,
        seq_len=seq_len,
        batch_size=batch_size,
        num_batches=steps,
        min_bytes=131072,
    )
    empty_cache(device)
    reset_peak_memory(device)

    with PowerMonitor(interval_sec=0.05) as monitor:
        report = train_fixed_steps(
            name,
            model,
            train_batches,
            eval_batches,
            device,
            optimizer,
            steps=steps,
            grad_accum_steps=1,
            clip_grad_norm=1.0,
            use_amp=use_amp,
        )
        synchronize_device(device)

    power = monitor.summary()
    peak_vram_mb = peak_memory_mb(device)
    energy_per_train_token = None
    if power["energy_joules"] is not None:
        energy_per_train_token = power["energy_joules"] / max(report.tokens, 1)

    return {
        "train_report": report.to_dict(),
        "power": power,
        "energy_j_per_train_token": energy_per_train_token,
        "peak_vram_mb": peak_vram_mb,
    }


def benchmark_and_measure_power(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    iterations: int,
) -> dict:
    with PowerMonitor(interval_sec=0.02) as monitor:
        throughput = benchmark_forward(
            model,
            input_ids,
            labels,
            device,
            iterations=iterations,
            warmup=2,
        )
        synchronize_device(device)
    power = monitor.summary()
    energy_per_token = None
    if power["energy_joules"] is not None:
        energy_per_token = power["energy_joules"] / max(throughput["tokens"], 1.0)
    return {
        **throughput,
        "power": power,
        "energy_j_per_forward_token": energy_per_token,
    }


def run(args: argparse.Namespace) -> dict:
    set_seed(args.seed)
    RECORDS.mkdir(parents=True, exist_ok=True)
    requested_device = "cpu" if args.cpu else args.device
    device = resolve_device(requested_device)
    tokenizer = ByteTokenizer()
    text = make_mixed_reasoning_corpus(seed=args.seed)
    prompts = make_task_prompts()
    max_new_tokens = args.max_new_tokens
    max_prompt_tokens = max(tokenizer.encode(item.prompt).numel() for item in prompts)
    model_max_seq_len = max(args.seq_len, max_prompt_tokens + max_new_tokens + 2)
    eval_batches = fixed_eval_batches(
        text,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        num_batches=4,
        min_bytes=65536,
    )

    results = {
        "environment": environment_report(),
        "device_used": str(device),
        "benchmark_scope": (
            "Small synthetic language/reasoning benchmark. This is useful for "
            "architecture comparison, not evidence of frontier or AGI behavior."
        ),
        "settings": {
            "seed": args.seed,
            "seq_len": args.seq_len,
            "model_max_seq_len": model_max_seq_len,
            "batch_size": args.batch_size,
            "steps": args.steps,
            "amp": not args.no_amp,
            "bench_iterations": args.bench_iterations,
            "max_new_tokens": max_new_tokens,
        },
        "models": {},
    }

    models = make_models(model_max_seq_len)
    results["parameter_match"] = parameter_match_report(
        models["parallel_cmf"],
        models["transformer"],
        tolerance=args.parameter_tolerance,
    )

    for name, model in models.items():
        start = time.perf_counter()
        train_data = train_and_measure(
            name,
            model,
            text,
            eval_batches,
            device,
            args.seq_len,
            args.batch_size,
            args.steps,
            use_amp=not args.no_amp,
        )
        final_eval_loss = evaluate_loss(model, eval_batches, device)
        prompt_eval = evaluate_prompt_accuracy(
            model,
            tokenizer,
            prompts,
            device,
            max_new_tokens=max_new_tokens,
        )
        candidate_eval = evaluate_candidate_accuracy(
            model,
            tokenizer,
            prompts,
            device,
            numeric_max=120,
        )
        bench_batch = next(
            make_lm_batches(
                text,
                seq_len=args.seq_len,
                batch_size=args.batch_size,
                num_batches=1,
                min_bytes=65536,
            )
        )
        throughput = benchmark_and_measure_power(
            model,
            bench_batch[0],
            bench_batch[1],
            device,
            iterations=args.bench_iterations,
        )
        elapsed = time.perf_counter() - start
        results["models"][name] = {
            "parameters": count_parameters(model),
            "final_eval_loss": final_eval_loss,
            "perplexity": math.exp(min(final_eval_loss, 20.0)),
            "prompt_eval": prompt_eval,
            "candidate_eval": candidate_eval,
            "throughput": throughput,
            "training": train_data,
            "total_elapsed_sec": elapsed,
        }

        row_path = RECORDS / f"{name}_prompt_rows.csv"
        with row_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["kind", "prompt", "expected", "generated", "answer", "correct"],
            )
            writer.writeheader()
            writer.writerows(prompt_eval["rows"])
        results["models"][name]["prompt_rows_csv"] = str(row_path)
        candidate_row_path = RECORDS / f"{name}_candidate_rows.csv"
        with candidate_row_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "kind",
                    "prompt",
                    "expected",
                    "best_answer",
                    "best_score",
                    "expected_rank",
                    "correct",
                ],
            )
            writer.writeheader()
            writer.writerows(candidate_eval["rows"])
        results["models"][name]["candidate_rows_csv"] = str(candidate_row_path)

    transformer = results["models"].get("transformer")
    cmf_candidates = {
        name: data for name, data in results["models"].items() if name.endswith("cmf")
    }
    cmf = None
    cmf_name = None
    if cmf_candidates:
        cmf_name, cmf = max(
            cmf_candidates.items(),
            key=lambda item: score_cmf_candidate(item[1], transformer),
        )
    comparison = {}
    if transformer and cmf:
        train_energy_ratio = (
            cmf["training"]["energy_j_per_train_token"]
            / max(transformer["training"]["energy_j_per_train_token"], 1e-12)
            if cmf["training"]["energy_j_per_train_token"] is not None
            and transformer["training"]["energy_j_per_train_token"] is not None
            else None
        )
        forward_energy_ratio = (
            cmf["throughput"]["energy_j_per_forward_token"]
            / max(transformer["throughput"]["energy_j_per_forward_token"], 1e-12)
            if cmf["throughput"]["energy_j_per_forward_token"] is not None
            and transformer["throughput"]["energy_j_per_forward_token"] is not None
            else None
        )
        comparison = {
            "selected_cmf": cmf_name,
            "selected_cmf_loss_div_transformer_loss": cmf["final_eval_loss"]
            / max(transformer["final_eval_loss"], 1e-12),
            "selected_cmf_accuracy_minus_transformer": cmf["prompt_eval"]["accuracy"]
            - transformer["prompt_eval"]["accuracy"],
            "selected_cmf_candidate_accuracy_minus_transformer": cmf["candidate_eval"][
                "accuracy"
            ]
            - transformer["candidate_eval"]["accuracy"],
            "selected_cmf_tokens_per_sec_div_transformer": cmf["throughput"]["tokens_per_sec"]
            / max(transformer["throughput"]["tokens_per_sec"], 1e-12),
            "selected_cmf_peak_vram_div_transformer": (
                cmf["training"]["peak_vram_mb"]
                / max(transformer["training"]["peak_vram_mb"], 1e-12)
                if cmf["training"]["peak_vram_mb"] is not None
                and transformer["training"]["peak_vram_mb"] is not None
                else None
            ),
            "selected_cmf_train_energy_per_token_div_transformer": train_energy_ratio,
            "selected_cmf_forward_energy_per_token_div_transformer": forward_energy_ratio,
        }
    results["comparison_vs_transformer"] = comparison
    results["claim"] = make_claim(results)

    write_json(RECORDS / "latest.json", results)
    write_markdown_report(RECORDS / "latest.md", "Quality Efficiency Run", results)
    return results


def make_claim(results: dict) -> dict:
    comparison = results.get("comparison_vs_transformer", {})
    if not comparison:
        return {"beats_transformer": False, "reason": "Transformer comparison missing."}
    parameter_match = results.get("parameter_match", {})
    parameters_matched = bool(parameter_match.get("matched", False))

    quality_win = (
        comparison["selected_cmf_loss_div_transformer_loss"] <= 1.0
        and comparison["selected_cmf_accuracy_minus_transformer"] >= 0.0
        and comparison["selected_cmf_candidate_accuracy_minus_transformer"] >= 0.0
    )
    efficiency_win = comparison["selected_cmf_tokens_per_sec_div_transformer"] >= 1.0
    train_energy_ratio = comparison.get("selected_cmf_train_energy_per_token_div_transformer")
    forward_energy_ratio = comparison.get("selected_cmf_forward_energy_per_token_div_transformer")
    train_energy_win = train_energy_ratio is not None and train_energy_ratio <= 0.1
    forward_energy_win = forward_energy_ratio is not None and forward_energy_ratio <= 0.1
    landslide = (
        parameters_matched
        and quality_win
        and efficiency_win
        and train_energy_win
        and forward_energy_win
    )

    if landslide:
        reason = "CMF matched/exceeded quality, was faster, and used <=10% train and inference energy per token on this small benchmark."
    elif not parameters_matched:
        reason = "Model parameter counts are not within tolerance, so no fair-comparison win is claimed."
    elif quality_win and efficiency_win:
        reason = "CMF matched/exceeded quality and throughput on this small benchmark, but did not prove a 10x energy win."
    elif quality_win:
        reason = "CMF matched/exceeded quality, but efficiency is not better."
    else:
        reason = "CMF did not yet beat the Transformer baseline on the combined quality gate."

    return {
        "beats_transformer": parameters_matched and quality_win and efficiency_win,
        "landslide_10x_energy": landslide,
        "parameters_matched": parameters_matched,
        "scope": results.get("benchmark_scope"),
        "reason": reason,
    }


def score_cmf_candidate(cmf: dict, transformer: dict | None) -> tuple:
    if transformer is None:
        return (0, cmf["throughput"]["tokens_per_sec"], -cmf["final_eval_loss"])
    quality_win = (
        cmf["final_eval_loss"] <= transformer["final_eval_loss"]
        and cmf["prompt_eval"]["accuracy"] >= transformer["prompt_eval"]["accuracy"]
        and cmf["candidate_eval"]["accuracy"] >= transformer["candidate_eval"]["accuracy"]
    )
    throughput_ratio = cmf["throughput"]["tokens_per_sec"] / max(
        transformer["throughput"]["tokens_per_sec"],
        1e-12,
    )
    forward_energy = cmf["throughput"].get("energy_j_per_forward_token")
    transformer_forward_energy = transformer["throughput"].get("energy_j_per_forward_token")
    if forward_energy is not None and transformer_forward_energy is not None:
        energy_score = transformer_forward_energy / max(forward_energy, 1e-12)
    else:
        energy_score = 0.0
    return (
        int(quality_win),
        throughput_ratio,
        energy_score,
        -cmf["final_eval_loss"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CMF quality/efficiency comparison.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--bench-iterations", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--parameter-tolerance", type=float, default=0.02)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
