import pytest

torch = pytest.importorskip("torch")

from cmf.generation import ChatState, encode_to_tensor, top_p_filter
from cmf.presets import PRESETS, estimate_cmf_parameters, get_preset
from cmf.data import ByteTokenizer


def test_infinity_8b_preset_is_8b_class():
    preset = get_preset("infinity-8b")
    params = estimate_cmf_parameters(preset.config, model_type=preset.model_type)
    assert 7_500_000_000 <= params <= 8_800_000_000
    assert preset.display_name.startswith("CMF Infinity")


def test_infinity_120m_preset_is_120m_class():
    preset = get_preset("infinity-0.12b")
    params = estimate_cmf_parameters(preset.config, model_type=preset.model_type)
    assert 115_000_000 <= params <= 125_000_000
    assert preset.model_type == "parallel_cmf"


def test_infinity_reasoning_120m_preset_is_120m_class():
    preset = get_preset("infinity-reasoning-0.12b")
    params = estimate_cmf_parameters(preset.config, model_type=preset.model_type)
    assert 115_000_000 <= params <= 125_000_000
    assert preset.model_type == "deliberative_cmf"
    assert preset.config.adaptive_thinking


def test_chat_state_renders_assistant_prompt():
    state = ChatState(system_prompt="Be precise.")
    state.add_user("hello")
    rendered = state.render()
    assert "System: Be precise." in rendered
    assert "User: hello" in rendered
    assert rendered.endswith("Assistant:")


def test_encode_to_tensor_with_byte_tokenizer():
    ids = encode_to_tensor(ByteTokenizer(), "abc")
    assert ids.dtype == torch.long
    assert ids.tolist() == [97, 98, 99]


def test_top_p_filter_keeps_shape():
    logits = torch.randn(2, 8)
    filtered = top_p_filter(logits, 0.8)
    assert filtered.shape == logits.shape
    assert torch.isfinite(filtered).any()
