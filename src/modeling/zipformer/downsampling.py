import copy
from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor, nn

from .encoder_layer import BypassModule
from .scaling import FloatLike


class DownsampledZipformer2Encoder(nn.Module):
    r"""
    DownsampledZipformer2Encoder is a zipformer encoder evaluated at a reduced frame rate,
    after convolutional downsampling, and then upsampled again at the output, and combined
    with the origin input, so that the output has the same shape as the input.
    """

    def __init__(self, encoder: nn.Module, dim: int, downsample: int, dropout: FloatLike) -> None:
        super(DownsampledZipformer2Encoder, self).__init__()
        self.downsample_factor = downsample
        self.downsample = SimpleDownsample(dim, downsample, dropout)
        self.num_layers = encoder.num_layers
        self.encoder = encoder
        self.upsample = SimpleUpsample(dim, downsample)
        self.out_combiner = BypassModule(dim, straight_through_rate=0)

    def forward(
        self,
        src: Tensor,
        chunk_size: int = -1,
        feature_mask: Union[Tensor, float] = 1.0,
        attn_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        r"""Downsample, go through encoder, upsample.

        Args:
            src: the sequence to the encoder (required): shape (seq_len, batch_size, embedding_dim).
            feature_mask: something that broadcasts with src, that we'll multiply `src`
               by at every layer: if a Tensor, likely of shape (seq_len, batch_size, embedding_dim)
            attn_mask: the attention mask, of shape (batch_size, seq_len, seq_len) or (seq_len, seq_len),
                 interpreted as (batch_size, tgt_seq_len, src_seq_len) or (tgt_seq_len, src_seq_len).
                 True means masked position. May be None.
            src_key_padding_mask:  the mask for padding, of shape (batch_size, seq_len); True means
                 masked position.  May be None.

        Returns: a Tensor with the same shape as src.
        """
        src_orig = src
        ds = self.downsample_factor
        src = self.downsample(src, src_key_padding_mask)
        if attn_mask is not None:
            attn_mask = attn_mask[::ds, ::ds]

        downsampled_padding_mask = None if src_key_padding_mask is None else src_key_padding_mask[..., ::ds]
        src = self.encoder(
            src,
            chunk_size=chunk_size // ds,
            feature_mask=feature_mask,
            attn_mask=attn_mask,
            src_key_padding_mask=downsampled_padding_mask,
        )
        src = self.upsample(src)
        # remove any extra frames that are not a multiple of downsample_factor
        src = src[: src_orig.shape[0]]

        return self.out_combiner(src_orig, src)

    def streaming_forward(
        self,
        src: Tensor,
        states: List[Tensor],
        left_context_len: int,
        src_key_padding_mask: Tensor,
    ) -> Tuple[Tensor, List[Tensor]]:
        r"""Downsample, go through encoder, upsample, in streaming forward mode.

        Args:
            src: the sequence to the encoder (required): shape (seq_len, batch_size, embedding_dim).
            states: list of cached tensors of N encoder layers. For layer-i, states[i*6:(i+1)*6] is
              (cached_key, cached_nonlin_attn, cached_val1, cached_val2, cached_conv1, cached_conv2).
            left_context_len: Number of left context frames.
            src_key_padding_mask: the mask for padding, of shape (batch_size, left_context_len+seq_len);
              True means masked position. May be None.

        Returns:
            - output, a Tensor with the same shape as src.
            - updated states
        """
        src_orig = src
        src = self.downsample(src)

        src, new_states = self.encoder.streaming_forward(
            src,
            states=states,
            left_context_len=left_context_len,
            src_key_padding_mask=src_key_padding_mask,
        )
        src = self.upsample(src)
        # remove any extra frames that are not a multiple of downsample_factor
        src = src[: src_orig.shape[0]]

        return self.out_combiner(src_orig, src), new_states


class SimpleDownsample(torch.nn.Module):
    """
    Does downsampling with attention, by weighted sum, and a projection..
    """

    def __init__(
        self,
        channels: int,
        downsample: int,
        dropout: FloatLike,
    ) -> None:
        super(SimpleDownsample, self).__init__()

        self.bias = nn.Parameter(torch.zeros(downsample))

        self.name = None  # will be set from training code
        self.dropout = copy.deepcopy(dropout)

        self.downsample = downsample

    def forward(
        self,
        src: Tensor,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        x: (seq_len, batch_size, in_channels)
        Returns a tensor of shape
           ( (seq_len+downsample-1)//downsample, batch_size, channels)
        """
        (seq_len, batch_size, in_channels) = src.shape
        ds = self.downsample
        d_seq_len = (seq_len + ds - 1) // ds

        # ---------------------------------------------------------
        # Match unpadded inference by extending each sequence's last valid frame.
        # ---------------------------------------------------------
        if src_key_padding_mask is not None:
            lengths = (~src_key_padding_mask).sum(dim=-1)
            last_valid_indices = lengths.sub(1).view(1, batch_size, 1)
            last_valid = src.gather(
                0,
                last_valid_indices.expand(1, batch_size, in_channels),
            )
            src = torch.where(
                src_key_padding_mask.transpose(0, 1).unsqueeze(-1),
                last_valid,
                src,
            )

        # Pad to an exact multiple of self.downsample
        # right-pad src, repeating the last element.
        pad = d_seq_len * ds - seq_len
        src_extra = src[src.shape[0] - 1 :].expand(pad, src.shape[1], src.shape[2])
        src = torch.cat((src, src_extra), dim=0)
        assert src.shape[0] == d_seq_len * ds

        src = src.reshape(d_seq_len, ds, batch_size, in_channels)

        weights = self.bias.softmax(dim=0)
        # weights: (downsample, 1, 1)
        weights = weights.unsqueeze(-1).unsqueeze(-1)

        # ans1 is the first `in_channels` channels of the output
        ans = (src * weights).sum(dim=1)

        return ans


class SimpleUpsample(torch.nn.Module):
    """
    A very simple form of upsampling that mostly just repeats the input, but
    also adds a position-specific bias.
    """

    def __init__(self, num_channels: int, upsample: int) -> None:
        super(SimpleUpsample, self).__init__()
        self.upsample = upsample

    def forward(self, src: Tensor) -> Tensor:
        """
        x: (seq_len, batch_size, num_channels)
        Returns a tensor of shape
           ( (seq_len*upsample), batch_size, num_channels)
        """
        upsample = self.upsample
        (seq_len, batch_size, num_channels) = src.shape
        src = src.unsqueeze(1).expand(seq_len, upsample, batch_size, num_channels)
        src = src.reshape(seq_len * upsample, batch_size, num_channels)
        return src
