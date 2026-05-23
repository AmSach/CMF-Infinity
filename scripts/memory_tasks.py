"""
Phase 1 — Synthetic streaming memory tasks.

Natural language hides shortcuts (n-gram overlap, position bias, frequency).
These tasks make shortcuts impossible by construction.

Vocabulary
----------
0–9   digit distractors
10–35 key tokens (26 keys)
36–61 value tokens (26 values)
62    SEP
63    QUERY
64    UPDATE (for object permanence)
65    PAD

All tasks return {input_ids, labels} where labels is -100 everywhere
except at the query answer position — so loss = retrieval accuracy directly.

Retention degradation curve (Phase 1.2):
    accuracy = f(gap_length)
    Baseline (memoryless): 1/26 ≈ 3.8%

Capacity law (Phase 1.4):
    accuracy = f(n_bindings, num_slots, d_model)

Run measure_retention_curve() and measure_capacity_curve() to get these.
"""

from __future__ import annotations
import random
from typing import Optional
import torch
from torch import Tensor
from torch.utils.data import Dataset

PAD   = 65
QUERY = 63
SEP   = 62
UPD   = 64
N_KEYS = 26
N_VALS = 26
KEY_OFF = 10
VAL_OFF = 36
VOCAB_SIZE = 66


def key_tok(k: int) -> int:  return KEY_OFF + (k % N_KEYS)
def val_tok(v: int) -> int:  return VAL_OFF + (v % N_VALS)
def rand_dist(n: int, rng: random.Random) -> list[int]:
    return [rng.randint(0, 9) for _ in range(n)]


# ─────────────────────────────────────────────────────────────────────────────

class KeyDoorDataset(Dataset):
    """
    Sequence: [K V SEP ...distractors(gap)... QUERY] → V

    gap is sampled from gap_lengths each call so one dataset covers the
    full retention curve.
    """
    def __init__(self, n_samples: int,
                 gap_lengths: Optional[list[int]] = None,
                 seed: int = 42):
        self.n = n_samples
        self.gaps = gap_lengths or [16, 64, 128, 512, 1024]
        self.rng  = random.Random(seed)

    def __len__(self):  return self.n

    def __getitem__(self, idx: int) -> dict:
        gap = self.rng.choice(self.gaps)
        k   = self.rng.randint(0, N_KEYS - 1)
        v   = self.rng.randint(0, N_VALS - 1)
        toks = [key_tok(k), val_tok(v), SEP] + rand_dist(gap, self.rng) + [QUERY]
        T    = len(toks)
        ids  = torch.tensor(toks, dtype=torch.long)
        lbl  = torch.full((T,), -100, dtype=torch.long)
        lbl[-1] = val_tok(v)
        return {"input_ids": ids, "labels": lbl,
                "gap": gap, "key": k, "value": v, "query_pos": T - 1}


class MultiBindingDataset(Dataset):
    """
    Store K key→value pairs, then query one.
    Tests: can the latent space hold K independent bindings?

    Sequence: [K1 V1 K2 V2 ... KK VK SEP ...gap... QUERY Ki] → Vi
    """
    def __init__(self, n_samples: int,
                 k_list: Optional[list[int]] = None,
                 gap: int = 32, seed: int = 42):
        self.n    = n_samples
        self.k_list = k_list or [2, 4, 8, 16]
        self.gap  = gap
        self.rng  = random.Random(seed)

    def __len__(self):  return self.n

    def __getitem__(self, idx: int) -> dict:
        K    = self.rng.choice(self.k_list)
        keys = self.rng.sample(range(N_KEYS), min(K, N_KEYS))
        vals = [self.rng.randint(0, N_VALS - 1) for _ in keys]
        bmap = dict(zip(keys, vals))

        toks: list[int] = []
        for kk, vv in zip(keys, vals):
            toks += [key_tok(kk), val_tok(vv)]
        toks += [SEP]
        toks += rand_dist(self.gap, self.rng)

        qi = self.rng.randint(0, len(keys) - 1)
        qk, qv = keys[qi], bmap[keys[qi]]
        toks += [QUERY, key_tok(qk)]

        T   = len(toks)
        ids = torch.tensor(toks, dtype=torch.long)
        lbl = torch.full((T,), -100, dtype=torch.long)
        lbl[-1] = val_tok(qv)
        return {"input_ids": ids, "labels": lbl,
                "n_bindings": K, "query_key": qk, "correct_value": qv}


class ObjectPermanenceDataset(Dataset):
    """
    A binding is updated mid-stream; model must return the LATEST value.

    Sequence: [K V_old SEP ...gap... UPD K V_new SEP ...gap... QUERY] → V_new
    Tests: does the model track updates or anchor to first binding?
    """
    def __init__(self, n_samples: int, gap: int = 32, seed: int = 42):
        self.n   = n_samples
        self.gap = gap
        self.rng = random.Random(seed)

    def __len__(self):  return self.n

    def __getitem__(self, idx: int) -> dict:
        k     = self.rng.randint(0, N_KEYS - 1)
        v_old = self.rng.randint(0, N_VALS - 1)
        v_new = self.rng.randint(0, N_VALS - 1)
        while v_new == v_old:
            v_new = self.rng.randint(0, N_VALS - 1)

        toks  = ([key_tok(k), val_tok(v_old), SEP]
                 + rand_dist(self.gap, self.rng)
                 + [UPD, key_tok(k), val_tok(v_new), SEP]
                 + rand_dist(self.gap, self.rng)
                 + [QUERY])
        T     = len(toks)
        ids   = torch.tensor(toks, dtype=torch.long)
        lbl   = torch.full((T,), -100, dtype=torch.long)
        lbl[-1] = val_tok(v_new)
        return {"input_ids": ids, "labels": lbl,
                "v_old": v_old, "v_new": v_new, "key": k}


# ─────────────────────────────────────────────────────────────────────────────
# Measurement helpers
# ─────────────────────────────────────────────────────────────────────────────

def _collate(samples: list[dict]) -> dict:
    """Pad a list of variable-length samples to the same length."""
    max_len = max(s["input_ids"].size(0) for s in samples)
    ids_out = torch.full((len(samples), max_len), PAD, dtype=torch.long)
    lbl_out = torch.full((len(samples), max_len), -100, dtype=torch.long)
    for i, s in enumerate(samples):
        L = s["input_ids"].size(0)
        ids_out[i, :L] = s["input_ids"]
        lbl_out[i, :L] = s["labels"]
    return {"input_ids": ids_out, "labels": lbl_out}


@torch.no_grad()
def measure_retention_curve(
    model,
    gap_lengths: list[int],
    n_per_gap: int = 200,
    device: str = "cpu",
) -> dict[int, float]:
    """
    Phase 1.2 measurement: accuracy vs gap length.
    Returns {gap: accuracy}.
    Baseline (random): 1/26 ≈ 0.038.
    """
    model.eval()
    results: dict[int, float] = {}
    for gap in gap_lengths:
        ds = KeyDoorDataset(n_per_gap, gap_lengths=[gap])
        correct = 0
        for idx in range(len(ds)):
            s       = ds[idx]
            ids     = s["input_ids"].unsqueeze(0).to(device)
            qpos    = s["query_pos"]
            correct_tok = s["labels"][qpos].item()
            out     = model(ids)
            pred    = out["logits"][0, qpos].argmax().item()
            correct += int(pred == correct_tok)
        results[gap] = correct / n_per_gap
        print(f"  gap={gap:5d}  acc={results[gap]:.3f}  (baseline=0.038)")
    return results


@torch.no_grad()
def measure_capacity_curve(
    model,
    k_list: list[int],
    n_per_k: int = 200,
    gap: int = 32,
    device: str = "cpu",
) -> dict[int, float]:
    """
    Phase 1.4 measurement: accuracy vs number of simultaneous bindings.
    Returns {n_bindings: accuracy}.
    """
    model.eval()
    results: dict[int, float] = {}
    for k in k_list:
        ds = MultiBindingDataset(n_per_k, k_list=[k], gap=gap)
        correct = 0
        for idx in range(len(ds)):
            s      = ds[idx]
            ids    = s["input_ids"].unsqueeze(0).to(device)
            T      = ids.size(1)
            correct_tok = val_tok(s["correct_value"])
            out    = model(ids)
            pred   = out["logits"][0, T - 1].argmax().item()
            correct += int(pred == correct_tok)
        results[k] = correct / n_per_k
        print(f"  n_bindings={k:3d}  acc={results[k]:.3f}")
    return results
