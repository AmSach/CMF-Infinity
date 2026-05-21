import pytest

torch = pytest.importorskip("torch")

from cmf import CMFConfig, DeliberativeContinuousMeaningField, ParallelContinuousMeaningField
from cmf.checkpointing import load_model_package, save_model_package
from cmf.data import ByteTokenizer


def test_model_package_round_trip(tmp_path):
    config = CMFConfig(vocab_size=32, d_model=8, hidden_dim=16, num_layers=1)
    model = ParallelContinuousMeaningField(config)
    tokenizer = ByteTokenizer(vocab_size=256)
    path = tmp_path / "tiny.package.pt"

    save_model_package(
        path,
        model,
        model_type="parallel_cmf",
        config=config,
        tokenizer=tokenizer,
        training={"steps": 0},
    )
    loaded, loaded_tokenizer, payload = load_model_package(path, device=torch.device("cpu"))

    assert payload["format"] == "cmf.model_package.v1"
    assert loaded.config.vocab_size == config.vocab_size
    assert loaded_tokenizer.decode(loaded_tokenizer.encode("abc")) == "abc"


def test_raw_state_dict_is_not_loaded_as_package(tmp_path):
    path = tmp_path / "raw.pt"
    torch.save({"embedding.weight": torch.zeros(2, 2)}, path)

    with pytest.raises(ValueError, match="not a CMF model package"):
        load_model_package(path, device=torch.device("cpu"))


def test_deliberative_model_package_round_trip(tmp_path):
    config = CMFConfig(
        vocab_size=32,
        d_model=8,
        hidden_dim=16,
        num_layers=1,
        thinking_steps=2,
        adaptive_thinking=True,
        max_thinking_steps=2,
    )
    model = DeliberativeContinuousMeaningField(config)
    tokenizer = ByteTokenizer(vocab_size=256)
    path = tmp_path / "tiny_deliberative.package.pt"

    save_model_package(
        path,
        model,
        model_type="deliberative_cmf",
        config=config,
        tokenizer=tokenizer,
        training={"steps": 0},
    )
    loaded, _loaded_tokenizer, payload = load_model_package(path, device=torch.device("cpu"))

    assert payload["model_type"] == "deliberative_cmf"
    assert isinstance(loaded, DeliberativeContinuousMeaningField)
