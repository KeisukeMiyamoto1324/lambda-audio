import torch
from transformers.models.lfm2.configuration_lfm2 import Lfm2Config
from transformers.models.lfm2.modeling_lfm2 import Lfm2ForCausalLM

from src.modeling.lfm2_decoder import (
    AudioLfm2DecoderLayer,
    AudioLfm2ForCausalLM,
    add_cross_attention,
)


def test_only_full_attention_layers_are_replaced() -> None:
    config = Lfm2Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        layer_types=["conv", "full_attention", "conv"],
    )
    model = Lfm2ForCausalLM(config)

    add_cross_attention(model)

    assert not isinstance(model.model.layers[0], AudioLfm2DecoderLayer)
    assert isinstance(model.model.layers[1], AudioLfm2DecoderLayer)
    assert not isinstance(model.model.layers[2], AudioLfm2DecoderLayer)


def test_decoder_returns_logits_without_labels() -> None:
    config = Lfm2Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        layer_types=["full_attention"],
    )
    model = AudioLfm2ForCausalLM(config)
    add_cross_attention(model)
    audio_features = torch.randn(2, 4, config.hidden_size)

    output = model(
        input_ids=torch.tensor([[1, 2, 3], [1, 4, 5]]),
        audio_key_values=model.project_audio_key_values(audio_features),
        audio_padding_mask=torch.zeros(2, 4, dtype=torch.bool),
    )

    assert output.loss is None
    assert output.logits is not None
    assert output.logits.shape == (2, 3, config.vocab_size)
