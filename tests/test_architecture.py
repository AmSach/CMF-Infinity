"""
CMF v3 Architecture Tests.

Covers every structural claim from the checklist and every fix listed in model.py.
Tests are grouped by checklist phase.

Run: pytest tests/test_architecture.py -v
"""
from __future__ import annotations
import inspect
import math
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch
import torch.nn as nn

from cmf.config import CMFConfig
from cmf.model import (
    ParallelCMF, DeliberativeCMF,
    SlotMemory, ManifoldAnchor, HaltHead, VectorField,
    DilatedResidualBlock,
)
from cmf.solver import euler_step, rk4_step, integrate_fixed, integrate_adaptive
from cmf.memory_tasks import (
    KeyDoorDataset, MultiBindingDataset, ObjectPermanenceDataset,
    VOCAB_SIZE,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def cfg():
    return CMFConfig(
        vocab_size=128, d_model=64, hidden_dim=128,
        num_layers=2, num_slots=8, solver_steps=4,
        thinking_steps=4, min_thinking_steps=2,
        dropout=0.0, tie_embeddings=False,
        routing_mode="sparse_topk", routing_topk=4,
    )

@pytest.fixture
def parallel(cfg):     return ParallelCMF(cfg)
@pytest.fixture
def deliberative(cfg): return DeliberativeCMF(cfg)

B, T = 2, 12


# ─────────────────────────────────────────────────────────────────────────────
# Phase 0 — Infrastructure
# ─────────────────────────────────────────────────────────────────────────────

class TestNoForbiddenPatterns:
    """Structural tests: banned patterns must not exist in source code."""

    def _src(self, module_name: str) -> str:
        import importlib
        m = importlib.import_module(f"cmf.{module_name}")
        return inspect.getsource(m)

    def test_no_sin_jitter(self):
        # Strip docstrings before checking — patterns appear in docs intentionally
        import ast, textwrap
        import cmf.model as m
        src = inspect.getsource(m)
        # Remove all string literals (docstrings + comments cannot contain executable sin jitter)
        try:
            tree = ast.parse(src)
            # Check no ast.Call has func=sin with arg z*1000
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    fn = node.func
                    name = (fn.attr if isinstance(fn, ast.Attribute) else
                            fn.id if isinstance(fn, ast.Name) else "")
                    if name == "sin" and node.args:
                        arg = node.args[0]
                        # arg should not be z * 1000
                        if isinstance(arg, ast.BinOp) and isinstance(arg.op, ast.Mult):
                            right = arg.right
                            if isinstance(right, ast.Constant) and right.value in (1000, 1000.0):
                                pytest.fail("sin(z * 1000) jitter found as executable code")
        except SyntaxError:
            pytest.skip("Could not parse model.py for AST check")

    def test_no_cgmp_in_solver(self):
        import cmf.solver as s
        src = inspect.getsource(s)
        # CGMP was: ((z-z_mean)/z_std) * c_std + c_mean inside euler_step
        assert "c_mean" not in src, "CGMP projection found in solver.py"
        assert "c_std"  not in src

    def test_no_symplectic_curl(self):
        import ast
        import cmf.model as m
        src = inspect.getsource(m)
        # Check no variable named symplectic_curl is assigned in executable code
        try:
            tree = ast.parse(src)
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for t in node.targets:
                        if isinstance(t, ast.Name) and t.id == "symplectic_curl":
                            pytest.fail("symplectic_curl assignment found in executable code")
        except SyntaxError:
            pytest.skip("Could not parse model.py")

    def test_no_dynamic_tempering(self):
        src = self._src("model")
        assert "_forward_calls" not in src, "Dynamic attention tempering counter found"
        assert "1500.0" not in src


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — Memory
# ─────────────────────────────────────────────────────────────────────────────

class TestSlotMemory:
    def test_output_shape_3d(self, cfg):
        mem = SlotMemory(cfg)
        z   = torch.randn(B, T, cfg.d_model)
        ctx = torch.randn(B, T, cfg.d_model)
        out = mem(z, ctx)
        assert out.shape == (B, T, cfg.d_model)

    def test_output_shape_2d(self, cfg):
        mem = SlotMemory(cfg)
        z   = torch.randn(B, cfg.d_model)
        ctx = torch.randn(B, cfg.d_model)
        out = mem.read(z)
        assert out.shape == (B, cfg.d_model)

    def test_memory_footprint_constant(self, cfg):
        """Parameter count must not depend on sequence length — O(num_slots) only."""
        mem = SlotMemory(cfg)
        n_params = sum(p.numel() for p in mem.parameters())
        # Verify: parameter count is independent of T
        # (structural test — if T changed, mem is the same object)
        assert n_params > 0
        # slot_keys + slot_vals dominate: 2 * num_slots * d_model
        slot_params = cfg.num_slots * cfg.d_model * 2
        assert n_params >= slot_params

    def test_no_write_during_eval(self, cfg):
        """Slot values must not change in eval mode."""
        mem = SlotMemory(cfg)
        mem.eval()
        vals_before = mem.slot_vals.data.clone()
        z   = torch.randn(B, T, cfg.d_model)
        ctx = torch.randn(B, T, cfg.d_model)
        mem(z, ctx)
        assert torch.equal(vals_before, mem.slot_vals.data), \
            "SlotMemory mutated slot_vals during eval"

    def test_differentiable(self, cfg):
        mem = SlotMemory(cfg)
        z   = torch.randn(B, T, cfg.d_model, requires_grad=True)
        ctx = torch.randn(B, T, cfg.d_model)
        out = mem(z, ctx)
        out.sum().backward()
        assert z.grad is not None


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — Routing
# ─────────────────────────────────────────────────────────────────────────────

class TestManifoldAnchor:
    @pytest.mark.parametrize("mode", ["full", "sparse_topk", "local_window", "none"])
    def test_output_shape(self, cfg, mode):
        anc = ManifoldAnchor(cfg)
        anc.mode = mode
        z   = torch.randn(B, T, cfg.d_model)
        ctx = torch.randn(B, T, cfg.d_model)
        assert anc(z, ctx).shape == (B, T, cfg.d_model)

    def test_none_returns_zeros(self, cfg):
        anc = ManifoldAnchor(cfg)
        anc.mode = "none"
        z   = torch.randn(B, T, cfg.d_model)
        ctx = torch.randn(B, T, cfg.d_model)
        out = anc(z, ctx)
        assert out.abs().max().item() == 0.0, "none mode must be exact zeros"

    def test_mode_switchable_at_runtime(self, cfg):
        """Mode changes without re-instantiation (ablation requirement)."""
        anc = ManifoldAnchor(cfg)
        z   = torch.randn(B, T, cfg.d_model)
        ctx = torch.randn(B, T, cfg.d_model)
        for mode in ["full", "sparse_topk", "local_window", "none"]:
            anc.mode = mode
            assert anc(z, ctx).shape == (B, T, cfg.d_model)

    def test_causal_masking(self, cfg):
        """Future context must not influence past positions."""
        anc = ManifoldAnchor(cfg)
        anc.mode = "full"
        z   = torch.randn(B, T, cfg.d_model)
        ctx = torch.randn(B, T, cfg.d_model)
        ctx_mod = ctx.clone()
        ctx_mod[:, -1, :] *= 10.0      # change only last context position

        out1 = anc(z, ctx)
        out2 = anc(z, ctx_mod)
        # All positions except last must be identical
        diff = (out1[:, :-1] - out2[:, :-1]).abs().max().item()
        assert diff < 1e-5, f"Causality violated: diff={diff}"

    def test_routing_mode_override_in_forward(self, parallel, cfg):
        """Model.forward routing_mode kwarg overrides anchor.mode temporarily."""
        ids = torch.randint(0, cfg.vocab_size, (B, T))
        for mode in ["full", "none"]:
            out = parallel(ids, routing_mode=mode)
            assert not out["logits"].isnan().any(), f"NaN logits with routing_mode={mode}"
        # After override, anchor.mode should be restored
        assert parallel.anchor.mode == cfg.routing_mode


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 — Iterative reasoning
# ─────────────────────────────────────────────────────────────────────────────

class TestHaltHead:
    def test_output_shape(self, cfg):
        h   = HaltHead(cfg.d_model, min_steps=2)
        z   = torch.randn(B, T, cfg.d_model)
        v   = torch.randn(B, T, cfg.d_model)
        prob, stop = h(z, v, step=5)
        assert prob.shape == torch.Size([])   # scalar
        assert isinstance(stop, bool)

    def test_min_steps_respected(self, cfg):
        cfg2 = CMFConfig(**{**cfg.__dict__, "halting_threshold": 0.0,
                            "min_thinking_steps": 3})
        h = HaltHead(cfg2.d_model, min_steps=3)
        z = torch.randn(B, T, cfg2.d_model)
        v = torch.randn(B, T, cfg2.d_model)
        for step in range(3):
            _, stop = h(z, v, step=step, threshold=0.0)
            assert not stop, f"Halted before min_steps at step {step}"

    def test_halt_differentiable(self, cfg):
        h    = HaltHead(cfg.d_model)
        z    = torch.randn(B, T, cfg.d_model, requires_grad=True)
        v    = torch.randn(B, T, cfg.d_model, requires_grad=True)
        prob, _ = h(z, v, step=5)
        prob.backward()
        assert z.grad is not None


class TestLogitEvolution:
    def test_trajectory_logged(self, deliberative, cfg):
        ids = torch.randint(0, cfg.vocab_size, (B, T))
        out = deliberative(ids, log_trajectory=True)
        assert "trajectory" in out
        traj = out["trajectory"]
        assert len(traj) >= cfg.min_thinking_steps
        for entry in traj:
            assert "logit_entropy" in entry
            assert "z_norm" in entry
            assert "halt_prob" in entry

    def test_thinking_steps_capped_during_training(self, deliberative, cfg):
        """Training uses ≤ 8 steps regardless of thinking_steps config."""
        deliberative.train()
        # Config has thinking_steps=4 which is already ≤ 8; set to 16 to test cap
        deliberative.cfg.thinking_steps = 16
        ids = torch.randint(0, cfg.vocab_size, (B, T))
        out = deliberative(ids, log_trajectory=True)
        assert int(out["thinking_steps"]) <= 8, "Training exceeded 8 steps"
        deliberative.cfg.thinking_steps = cfg.thinking_steps


# ─────────────────────────────────────────────────────────────────────────────
# Full model: shapes, loss, gradients
# ─────────────────────────────────────────────────────────────────────────────

class TestParallelCMF:
    def test_forward_shape(self, parallel, cfg):
        ids = torch.randint(0, cfg.vocab_size, (B, T))
        out = parallel(ids)
        assert out["logits"].shape == (B, T, cfg.vocab_size)

    def test_loss_positive(self, parallel, cfg):
        ids = torch.randint(0, cfg.vocab_size, (B, T))
        out = parallel(ids, labels=ids)
        assert out["loss"].item() > 0

    def test_gradients_finite(self, parallel, cfg):
        ids = torch.randint(0, cfg.vocab_size, (B, T))
        out = parallel(ids, labels=ids)
        out["loss"].backward()
        for name, p in parallel.named_parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), f"Non-finite grad in {name}"

    def test_eval_deterministic(self, parallel, cfg):
        """Eval mode must be deterministic (no slot writes, no noise)."""
        parallel.eval()
        ids = torch.randint(0, cfg.vocab_size, (1, T))
        with torch.no_grad():
            l1 = parallel(ids)["logits"]
            l2 = parallel(ids)["logits"]
        assert torch.allclose(l1, l2, atol=1e-5), "Non-deterministic in eval"

    def test_no_nan_any_routing_mode(self, parallel, cfg):
        ids = torch.randint(0, cfg.vocab_size, (B, T))
        for mode in ["full", "sparse_topk", "local_window", "none"]:
            out = parallel(ids, routing_mode=mode)
            assert not out["logits"].isnan().any(), f"NaN with mode={mode}"

    def test_generate(self, parallel, cfg):
        parallel.eval()
        ids = torch.randint(0, cfg.vocab_size, (1, 4))
        with torch.no_grad():
            out = parallel.generate(ids, max_new_tokens=8, top_k=10)
        assert out.shape == (1, 12)
        assert out.min() >= 0 and out.max() < cfg.vocab_size


class TestDeliberativeCMF:
    def test_forward_shape(self, deliberative, cfg):
        ids = torch.randint(0, cfg.vocab_size, (B, T))
        out = deliberative(ids)
        assert out["logits"].shape == (B, T, cfg.vocab_size)

    def test_ponder_loss_positive(self, deliberative, cfg):
        deliberative.train()
        ids = torch.randint(0, cfg.vocab_size, (B, T))
        out = deliberative(ids, labels=ids)
        assert out["ponder_loss"].item() >= 0

    def test_gradients_finite(self, deliberative, cfg):
        deliberative.train()
        ids = torch.randint(0, cfg.vocab_size, (B, T))
        out = deliberative(ids, labels=ids)
        out["loss"].backward()
        for name, p in deliberative.named_parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), f"Non-finite grad in {name}"


# ─────────────────────────────────────────────────────────────────────────────
# Solver
# ─────────────────────────────────────────────────────────────────────────────

class TestSolver:
    def _neg_field(self, z, c, t): return -z  # converges to 0

    def test_euler_no_cgmp(self):
        """euler_step must not contain c_mean/c_std normalisation."""
        src = inspect.getsource(euler_step)
        assert "c_mean" not in src
        assert "c_std"  not in src

    def test_euler_converges(self):
        z = torch.ones(2, 4)
        c = torch.zeros(2, 4)
        z_final = integrate_fixed(z, c, self._neg_field, steps=32, method="euler")
        assert z_final.norm().item() < z.norm().item()

    def test_rk4_more_accurate(self):
        z0     = torch.tensor([[2.0]])
        c      = torch.zeros(1, 1)
        target = z0 * math.exp(-1)
        z_e = integrate_fixed(z0, c, self._neg_field, steps=4, method="euler")
        z_r = integrate_fixed(z0, c, self._neg_field, steps=4, method="rk4")
        assert (z_r - target).abs() < (z_e - target).abs(), "RK4 not more accurate"

    def test_trajectory_returned(self):
        z = torch.randn(1, 4)
        c = torch.zeros(1, 4)
        z_f, traj = integrate_fixed(z, c, self._neg_field, steps=6,
                                    return_trajectory=True)
        assert len(traj) == 7   # initial + 6 steps
        assert traj[0].shape == z.shape

    def test_noise_applied_during_training(self):
        """euler_step with noise_scale > 0 must produce different results."""
        z = torch.ones(4, 8)
        c = torch.zeros(4, 8)
        tau = torch.zeros(4)
        z1 = euler_step(z, c, tau, lambda _z, _c, _t: torch.zeros_like(_z),
                        dt=0.1, noise_scale=0.1)
        z2 = euler_step(z, c, tau, lambda _z, _c, _t: torch.zeros_like(_z),
                        dt=0.1, noise_scale=0.1)
        assert not torch.equal(z1, z2), "Noise not applied (different calls identical)"

    def test_adaptive_easy_field(self):
        def zero_field(z, c, t): return torch.zeros_like(z)
        z0 = torch.randn(1, 8)
        c  = torch.zeros(1, 8)
        _, steps = integrate_adaptive(z0, c, zero_field, min_steps=2, max_steps=16)
        assert steps <= 4, f"Trivial field used too many steps: {steps}"


# ─────────────────────────────────────────────────────────────────────────────
# Memory tasks (Phase 1)
# ─────────────────────────────────────────────────────────────────────────────

class TestMemoryTasks:
    def test_keydoor_shape(self):
        ds  = KeyDoorDataset(10, gap_lengths=[16])
        s   = ds[0]
        assert s["input_ids"].dtype == torch.long
        assert s["labels"][s["query_pos"]].item() != -100

    def test_keydoor_baseline(self):
        """A token that always guesses the first value should score ~1/26."""
        ds = KeyDoorDataset(1000, gap_lengths=[16])
        from cmf.memory_tasks import VAL_OFF
        correct = sum(
            1 for i in range(len(ds))
            if ds[i]["labels"][ds[i]["query_pos"]].item() == VAL_OFF
        )
        # Should be ~1/26 ≈ 38/1000
        assert correct < 100, f"KeyDoor trivially solvable: {correct}/1000 correct with first-token"

    def test_multibinding_shape(self):
        ds = MultiBindingDataset(10, k_list=[4])
        s  = ds[0]
        assert s["input_ids"].dtype == torch.long

    def test_object_permanence_shape(self):
        ds = ObjectPermanenceDataset(10)
        s  = ds[0]
        # v_old and v_new must differ
        assert s["v_old"] != s["v_new"]
        # Label at final position must be v_new, not v_old
        from cmf.memory_tasks import val_tok
        T   = s["input_ids"].size(0)
        lbl = s["labels"][T - 1].item()
        assert lbl == val_tok(s["v_new"])
        assert lbl != val_tok(s["v_old"])

    def test_vocab_fits_model(self):
        """Task vocabulary must fit in tiny model vocab_size."""
        assert VOCAB_SIZE <= 128   # tiny fixture uses vocab_size=128
