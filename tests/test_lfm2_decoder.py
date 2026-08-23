from transformers.models.lfm2.configuration_lfm2 import Lfm2Config

from lfm2_decoder import (
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
    model = AudioLfm2ForCausalLM(config)

    add_cross_attention(model)

    assert not isinstance(model.model.layers[0], AudioLfm2DecoderLayer)
    assert isinstance(model.model.layers[1], AudioLfm2DecoderLayer)
    assert not isinstance(model.model.layers[2], AudioLfm2DecoderLayer)
