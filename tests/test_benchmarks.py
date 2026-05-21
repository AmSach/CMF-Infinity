import pytest

torch = pytest.importorskip("torch")

from cmf.benchmarks import extract_answer, make_task_prompts, parameter_match_report, score_candidate
from cmf import CMFConfig, ParallelContinuousMeaningField
from cmf.baselines import TinyGPTLM
from cmf.data import ByteTokenizer


def test_extract_answer():
    assert extract_answer("Q: 2+3=? A: 5\n") == "5"
    assert extract_answer("Q: what is cuda? A: gpu") == "gpu"
    assert extract_answer("Q: what is cuda? A: \nQ: what is mercury? A: planet") == ""


def test_task_prompts_have_expected_answers():
    prompts = make_task_prompts()
    assert prompts
    assert all(item.prompt.endswith("A: ") for item in prompts)
    assert all(item.answer for item in prompts)


def test_score_candidate_returns_float():
    tokenizer = ByteTokenizer()
    model = ParallelContinuousMeaningField(
        CMFConfig(vocab_size=256, d_model=8, hidden_dim=16, num_layers=1)
    )
    score = score_candidate(model, tokenizer, "Q: 1+1=? A: ", "2", torch.device("cpu"))
    assert isinstance(score, float)


def test_parameter_match_report_flags_matched_models():
    cmf = ParallelContinuousMeaningField(
        CMFConfig(vocab_size=256, d_model=80, hidden_dim=160, num_layers=3, tie_embeddings=False)
    )
    gpt = TinyGPTLM(
        vocab_size=256,
        d_model=96,
        nhead=4,
        num_layers=5,
        hidden_dim=216,
        dropout=0.0,
        max_seq_len=128,
    )
    report = parameter_match_report(cmf, gpt, tolerance=0.02)
    assert report["matched"]
