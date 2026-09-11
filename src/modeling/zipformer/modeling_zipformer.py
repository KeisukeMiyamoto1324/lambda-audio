import os
from typing import Any

import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput, Wav2Vec2BaseModelOutput

from .configuration_zipformer import ZipformerConfig
from .scaling import ScheduledFloat
from .utils import GradMultiply
from .wav2vec2_module import ConvFeatureExtractionModel
from .zipformer import Zipformer2


class ZipformerPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = ZipformerConfig
    base_model_prefix = "zipformer"
    supports_gradient_checkpointing = False

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            if hasattr(module, "parametrizations"):
                nn.init.kaiming_normal_(module.parametrizations.weight.original0.data)
                nn.init.kaiming_normal_(module.parametrizations.weight.original1.data)
            nn.init.kaiming_normal_(module.weight.data)

        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)) and module.bias is not None:
            module.bias.data.zero_()


class ZipformerModel(ZipformerPreTrainedModel):
    def __init__(self, config: ZipformerConfig) -> None:
        super().__init__(config)
        feature_enc_layers = config.conv_feature_layers
        self.embed = feature_enc_layers[-1][0]

        self.feature_extractor = ConvFeatureExtractionModel(
            conv_layers=feature_enc_layers,
            dropout=0.0,
            mode=config.extractor_mode,
            conv_bias=config.conv_bias,
        )
        encoder_input_dim = config.encoder_dim[0]
        self.post_extract_proj = nn.Linear(self.embed, encoder_input_dim) if self.embed != encoder_input_dim else None

        self.dropout_input = nn.Dropout(config.dropout_input)
        self.dropout_features = nn.Dropout(config.dropout_features)

        self.feature_grad_mult = config.feature_grad_mult

        self.mask_emb = nn.Parameter(torch.FloatTensor(encoder_input_dim).uniform_())

        self.encoder = Zipformer2(
            output_downsampling_factor=1,
            downsampling_factor=config.downsampling_factor,
            num_encoder_layers=config.num_encoder_layers,
            encoder_dim=config.encoder_dim,
            encoder_unmasked_dim=config.encoder_unmasked_dim,
            query_head_dim=config.query_head_dim,
            pos_head_dim=config.pos_head_dim,
            value_head_dim=config.value_head_dim,
            pos_dim=config.pos_dim,
            num_heads=config.num_heads,
            feedforward_dim=config.feedforward_dim,
            cnn_module_kernel=config.cnn_module_kernel,
            dropout=ScheduledFloat((0.0, 0.3), (20000.0, 0.1)),
            warmup_batches=4000.0,
        )

        self.layer_norm = nn.LayerNorm(self.embed)
        self.post_init()

    @classmethod
    def from_icefall_checkpoint(
        cls,
        checkpoint_path: os.PathLike[str],
        **kwargs: Any,
    ) -> "ZipformerModel":
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        config = ZipformerConfig.from_icefall_checkpoint(checkpoint_path, **kwargs)
        model = cls(config)
        state_dict = checkpoint["model"]
        state_dict.pop("final_proj.weight")
        state_dict.pop("final_proj.bias")
        model.load_state_dict(state_dict, strict=True)
        return model

    def forward_features(
        self,
        source: torch.Tensor,
        padding_mask: torch.BoolTensor | None = None,
    ) -> torch.Tensor:
        if self.feature_grad_mult > 0:
            features = self.feature_extractor(source, padding_mask)
            if self.feature_grad_mult != 1.0:
                features = GradMultiply.apply(features, self.feature_grad_mult)
        else:
            with torch.no_grad():
                features = self.feature_extractor(source, padding_mask)
        return features

    def forward_padding_mask(
        self,
        features: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        output_lengths = (~padding_mask).sum(dim=-1)
        for _, kernel_size, stride in self.config.conv_feature_layers:
            output_lengths = (
                torch.div(
                    output_lengths - kernel_size,
                    stride,
                    rounding_mode="floor",
                )
                + 1
            )

        positions = torch.arange(features.size(1), device=features.device)
        return positions.unsqueeze(0) >= output_lengths.unsqueeze(1)

    def forward(
        self,
        input_values: torch.FloatTensor,
        padding_mask: torch.BoolTensor | None = None,
    ) -> Wav2Vec2BaseModelOutput:
        """output layer is 1-based"""
        features = self.forward_features(input_values, padding_mask)

        features = features.transpose(1, 2)
        features = self.layer_norm(features)
        if padding_mask is None:
            padding_mask = torch.zeros(
                features.shape[:2],
                dtype=torch.bool,
                device=features.device,
            )
        else:
            padding_mask = self.forward_padding_mask(features, padding_mask)

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)

        features = self.dropout_input(features)
        x = features

        # feature: (B, T, D), float
        # target: (B, T), long
        # x: (B, T, D), float -> (T, B, D), float
        # padding_mask: (B, T), bool
        # mask_indices: (B, T), bool
        x = x.transpose(0, 1)
        x, _ = self.encoder(
            x,
            (~padding_mask).sum(dim=-1),
            src_key_padding_mask=padding_mask,
        )
        x = x.transpose(0, 1)

        return Wav2Vec2BaseModelOutput(
            last_hidden_state=x,
            extract_features=features,
        )


class ZipformerForCTC(ZipformerPreTrainedModel):
    def __init__(self, config: ZipformerConfig) -> None:
        super().__init__(config)
        self.encoder = ZipformerModel(config)
        self.ctc_output = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(max(config.encoder_dim), config.vocab_size),
        )

    @classmethod
    def from_icefall_checkpoint(
        cls,
        checkpoint_path: os.PathLike[str],
        **kwargs: Any,
    ) -> "ZipformerForCTC":
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        config = ZipformerConfig.from_icefall_checkpoint(checkpoint_path, **kwargs)
        model = cls(config)
        state_dict = checkpoint["model"]
        state_dict.pop("encoder.final_proj.weight")
        state_dict.pop("encoder.final_proj.bias")
        model.load_state_dict(state_dict, strict=True)
        return model

    def forward(
        self,
        input_values: torch.FloatTensor,
        padding_mask: torch.BoolTensor | None = None,
        **kwargs: Any,
    ) -> CausalLMOutput:
        if padding_mask is None:
            padding_mask = torch.zeros_like(input_values, dtype=torch.bool)

        encoder_out = self.encoder(input_values, padding_mask)
        ctc_output = self.ctc_output(encoder_out.last_hidden_state)

        return CausalLMOutput(loss=None, logits=ctc_output, hidden_states=None, attentions=None)
