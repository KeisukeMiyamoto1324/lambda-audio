import copy
from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor, nn

from .attention import CompactRelPositionalEncoding
from .scaling import ScheduledFloat


class Zipformer2Encoder(nn.Module):
    r"""Zipformer2Encoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the Zipformer2EncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
       pos_dim: the dimension for the relative positional encoding

    Examples::
        >>> encoder_layer = Zipformer2EncoderLayer(embed_dim=512, nhead=8)
        >>> zipformer_encoder = Zipformer2Encoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = zipformer_encoder(src)
    """

    def __init__(
        self,
        encoder_layer: nn.Module,
        num_layers: int,
        pos_dim: int,
        dropout: float,
        warmup_begin: float,
        warmup_end: float,
        initial_layerdrop_rate: float = 0.5,
        final_layerdrop_rate: float = 0.05,
    ) -> None:
        super().__init__()
        self.encoder_pos = CompactRelPositionalEncoding(pos_dim, dropout_rate=0.15, length_factor=1.0)

        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for i in range(num_layers)])
        self.num_layers = num_layers

        assert 0 <= warmup_begin <= warmup_end

        delta = (1.0 / num_layers) * (warmup_end - warmup_begin)
        cur_begin = warmup_begin  # interpreted as a training batch index
        for i in range(num_layers):
            cur_end = cur_begin + delta
            self.layers[i].bypass.skip_rate = ScheduledFloat(
                (cur_begin, initial_layerdrop_rate),
                (cur_end, final_layerdrop_rate),
                default=0.0,
            )
            cur_begin = cur_end

    def forward(
        self,
        src: Tensor,
        chunk_size: int = -1,
        feature_mask: Union[Tensor, float] = 1.0,
        attn_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required): shape (seq_len, batch_size, embedding_dim).
            chunk_size: the number of frames per chunk, of >= 0; if -1, no chunking.
            feature_mask: something that broadcasts with src, that we'll multiply `src`
               by at every layer: if a Tensor, likely of shape (seq_len, batch_size, embedding_dim)
            attn_mask: the attention mask, of shape (batch_size, seq_len, seq_len) or (seq_len, seq_len),
                 interpreted as (batch_size, tgt_seq_len, src_seq_len) or (tgt_seq_len, src_seq_len).
                 True means masked position. May be None.
            src_key_padding_mask:  the mask for padding, of shape (batch_size, seq_len); True means
                 masked position.  May be None.

        Returns: a Tensor with the same shape as src.
        """
        pos_emb = self.encoder_pos(src)
        output = src

        if not torch.jit.is_scripting() and not torch.jit.is_tracing():
            output = output * feature_mask

        for i, mod in enumerate(self.layers):
            output = mod(
                output,
                pos_emb,
                chunk_size=chunk_size,
                attn_mask=attn_mask,
                src_key_padding_mask=src_key_padding_mask,
            )

            if not torch.jit.is_scripting() and not torch.jit.is_tracing():
                output = output * feature_mask

        return output

    def streaming_forward(
        self,
        src: Tensor,
        states: List[Tensor],
        left_context_len: int,
        src_key_padding_mask: Tensor,
    ) -> Tuple[Tensor, List[Tensor]]:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required): shape (seq_len, batch_size, embedding_dim).
            states: list of cached tensors of N encoder layers. For layer-i, states[i*6:(i+1)*6] is
              (cached_key, cached_nonlin_attn, cached_val1, cached_val2, cached_conv1, cached_conv2).
            left_context_len: Number of left context frames.
            src_key_padding_mask:  the mask for padding, of shape
              (batch_size, left_context_len + seq_len); True means masked position.
              May be None.

        Returns:
          - output, a Tensor with the same shape as src.
          - updated states
        """
        pos_emb = self.encoder_pos(src, left_context_len)
        output = src

        new_states = []
        for i, mod in enumerate(self.layers):
            (
                cached_key,
                cached_nonlin_attn,
                cached_val1,
                cached_val2,
                cached_conv1,
                cached_conv2,
            ) = states[i * 6 : (i + 1) * 6]
            (
                output,
                new_cached_key,
                new_cached_nonlin_attn,
                new_cached_val1,
                new_cached_val2,
                new_cached_conv1,
                new_cached_conv2,
            ) = mod.streaming_forward(
                output,
                pos_emb,
                cached_key=cached_key,
                cached_nonlin_attn=cached_nonlin_attn,
                cached_val1=cached_val1,
                cached_val2=cached_val2,
                cached_conv1=cached_conv1,
                cached_conv2=cached_conv2,
                left_context_len=left_context_len,
                src_key_padding_mask=src_key_padding_mask,
            )
            new_states += [
                new_cached_key,
                new_cached_nonlin_attn,
                new_cached_val1,
                new_cached_val2,
                new_cached_conv1,
                new_cached_conv2,
            ]

        return output, new_states
