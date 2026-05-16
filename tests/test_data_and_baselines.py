import pytest

torch = pytest.importorskip("torch")

from cmf.baselines import TemporalConvLM, TinyGPTLM, TinyTransformerLM
from cmf.data import ByteTokenizer, cyclic_lm_batches, repeated_corpus


def test_byte_tokenizer_handles_unicode_roundtrip():
    tokenizer = ByteTokenizer()
    text = "meaning field: नमस्ते"
    encoded = tokenizer.encode(text)
    assert encoded.max().item() < tokenizer.vocab_size
    assert tokenizer.decode(encoded) == text


def test_cyclic_batches_shapes():
    data = repeated_corpus("abcde", min_bytes=128)
    batch = next(cyclic_lm_batches(data, seq_len=8, batch_size=3, num_batches=1))
    assert batch[0].shape == (3, 8)
    assert batch[1].shape == (3, 8)


@pytest.mark.parametrize("model_cls", [TemporalConvLM, TinyTransformerLM, TinyGPTLM])
def test_baseline_forward_shapes(model_cls):
    model = model_cls(vocab_size=32, d_model=16, hidden_dim=32, num_layers=1)
    input_ids = torch.randint(0, 32, (2, 8))
    labels = torch.randint(0, 32, (2, 8))
    output = model(input_ids, labels=labels, return_states=True)
    assert output["logits"].shape == (2, 8, 32)
    assert output["states"].shape[:2] == (2, 8)
    assert output["loss"].ndim == 0
