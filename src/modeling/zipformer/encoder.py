import logging
import random
import warnings
from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor, nn

from .downsampling import DownsampledZipformer2Encoder, SimpleDownsample
from .encoder_layer import Zipformer2EncoderLayer
from .encoder_stack import Zipformer2Encoder
from .scaling import FloatLike, ScheduledFloat, convert_num_channels


class Zipformer2(nn.Module):
    """
    Args:

    Note: all "int or Tuple[int]" arguments below will be treated as lists of the same length
    as downsampling_factor if they are single ints or one-element tuples.  The length of
    downsampling_factor defines the number of stacks.

        output_downsampling_factor (int): how much to downsample at the output.  Note:
            we also downsample by a factor of 2 in the Conv2dSubsampling encoder.
            You should probably leave this at 2.
        downsampling_factor (Tuple[int]): downsampling factor for each encoder stack.
           Note: this is in addition to the downsampling factor of 2 that is applied in
           the frontend (self.encoder_embed).
        encoder_dim (Tuple[int]): embedding dimension of each of the encoder stacks, one per
           encoder stack.
        num_encoder_layers (int or Tuple[int])): number of encoder layers for each stack
        encoder_unmasked_dim (int or Tuple[int]): unmasked dimension in each of
            the encoder stacks for purposes of per-frame dropout (recommend 256 for
            now).
        query_head_dim (int or Tuple[int]): dimension of query and key per attention
           head: per stack, if a tuple..
        pos_head_dim (int or Tuple[int]): dimension of positional-encoding projection per
           attention head
        value_head_dim (int or Tuple[int]): dimension of value in each attention head
        num_heads: (int or Tuple[int]): number of heads in the self-attention mechanism.
              Must be at least 4.
        feedforward_dim (int or Tuple[int]): hidden dimension in feedforward modules
        cnn_module_kernel (int or Tuple[int])): Kernel size of convolution module

        pos_dim (int): the dimension of each positional-encoding vector prior to projection,
            e.g. 128.

        dropout (float): dropout rate
        warmup_batches (float): number of batches to warm up over; this controls
          dropout of encoder layers.
        causal (bool): if True, support chunkwise causal convolution.  This should
          not hurt WER as no modeling power is lost, but the convolution modules will be
          slightly slower and use more memory.  Enables use of the chunk_size and
          left_context_chunks options in forward(), which simulates streaming
          decoding.
        chunk_size: (list of int): only set this to other than [-1] if causal;
           the chunk size will be randomly chosen from this list.  -1 means no chunking.
        left_context_frames: (list of int): determines the number of left-
           context chunks for causal training; will be rounded to a number of
           chunks.  Must not be less than cnn_module_kernel (after factoring in
           rounding and downsampling); an error will be thrown if this is violated.
    """

    def __init__(
        self,
        output_downsampling_factor: int = 2,
        downsampling_factor: Tuple[int] = (2, 4),
        encoder_dim: Union[int, Tuple[int]] = 384,
        num_encoder_layers: Union[int, Tuple[int]] = 4,
        encoder_unmasked_dim: Union[int, Tuple[int]] = 256,
        query_head_dim: Union[int, Tuple[int]] = 24,
        pos_head_dim: Union[int, Tuple[int]] = 4,
        value_head_dim: Union[int, Tuple[int]] = 12,
        num_heads: Union[int, Tuple[int]] = 8,
        feedforward_dim: Union[int, Tuple[int]] = 1536,
        cnn_module_kernel: Union[int, Tuple[int]] = 31,
        pos_dim: int = 192,
        dropout: FloatLike = None,  # see code below for default
        warmup_batches: float = 4000.0,
        causal: bool = False,
        chunk_size: Tuple[int] = (-1,),
        left_context_frames: Tuple[int] = (-1,),
    ) -> None:
        super(Zipformer2, self).__init__()

        if dropout is None:
            dropout = ScheduledFloat((0.0, 0.3), (20000.0, 0.1))

        def _to_tuple(x: int | Tuple[int]) -> Tuple[int]:
            """Converts a single int or a 1-tuple of an int to a tuple with the same length
            as downsampling_factor"""
            if isinstance(x, int):
                x = (x,)
            if len(x) == 1:
                x = x * len(downsampling_factor)
            else:
                assert len(x) == len(downsampling_factor) and isinstance(x[0], int)
            return x

        self.output_downsampling_factor = output_downsampling_factor
        self.downsampling_factor = downsampling_factor
        self.encoder_dim = encoder_dim
        self.encoder_unmasked_dim = encoder_unmasked_dim
        num_encoder_layers = num_encoder_layers
        self.num_encoder_layers = num_encoder_layers
        self.query_head_dim = query_head_dim = _to_tuple(query_head_dim)
        self.value_head_dim = value_head_dim = _to_tuple(value_head_dim)
        pos_head_dim = _to_tuple(pos_head_dim)
        self.num_heads = num_heads
        feedforward_dim = feedforward_dim
        self.cnn_module_kernel = cnn_module_kernel

        self.causal = causal
        self.chunk_size = chunk_size
        self.left_context_frames = left_context_frames

        for u, d in zip(encoder_unmasked_dim, encoder_dim):
            assert u <= d

        # each one will be Zipformer2Encoder or DownsampledZipformer2Encoder
        encoders = []

        num_encoders = len(downsampling_factor)
        for i in range(num_encoders):
            encoder_layer = Zipformer2EncoderLayer(
                embed_dim=encoder_dim[i],
                pos_dim=pos_dim,
                num_heads=num_heads[i],
                query_head_dim=query_head_dim[i],
                pos_head_dim=pos_head_dim[i],
                value_head_dim=value_head_dim[i],
                feedforward_dim=feedforward_dim[i],
                dropout=dropout,
                cnn_module_kernel=cnn_module_kernel[i],
                causal=causal,
            )

            # For the segment of the warmup period, we let the Conv2dSubsampling
            # layer learn something.  Then we start to warm up the other encoders.
            encoder = Zipformer2Encoder(
                encoder_layer,
                num_encoder_layers[i],
                pos_dim=pos_dim,
                dropout=dropout,
                warmup_begin=warmup_batches * (i + 1) / (num_encoders + 1),
                warmup_end=warmup_batches * (i + 2) / (num_encoders + 1),
                final_layerdrop_rate=0.035 * (downsampling_factor[i] ** 0.5),
            )

            if downsampling_factor[i] != 1:
                encoder = DownsampledZipformer2Encoder(
                    encoder,
                    dim=encoder_dim[i],
                    downsample=downsampling_factor[i],
                    dropout=dropout,
                )

            encoders.append(encoder)

        self.encoders = nn.ModuleList(encoders)

        self.downsample_output = SimpleDownsample(max(encoder_dim), downsample=output_downsampling_factor, dropout=dropout)

    def get_feature_masks(self, x: Tensor) -> Union[List[float], List[Tensor]]:
        """
        In eval mode, returns [1.0] * num_encoders; in training mode, returns a number of
        randomized feature masks, one per encoder.
        On e.g. 15% of frames, these masks will zero out all enocder dims larger than
        some supplied number, e.g. >256, so in effect on those frames we are using
        a smaller encoer dim.

        We generate the random masks at this level because we want the 2 masks to 'agree'
        all the way up the encoder stack. This will mean that the 1st mask will have
        mask values repeated self.zipformer_subsampling_factor times.

        Args:
           x: the embeddings (needed for the shape and dtype and device), of shape
             (1, batch_size, encoder_dims0)
        """
        num_encoders = len(self.encoder_dim)
        if not self.training:
            return [1.0] * num_encoders

        (num_frames0, batch_size, _encoder_dims0) = x.shape

        assert self.encoder_dim[0] == _encoder_dims0, (
            self.encoder_dim[0],
            _encoder_dims0,
        )

        feature_mask_dropout_prob = 0.125

        # mask1 shape: (1, batch_size, 1)
        mask1 = (torch.rand(1, batch_size, 1, device=x.device) > feature_mask_dropout_prob).to(x.dtype)

        # mask2 has additional sequences masked, about twice the number.
        mask2 = torch.logical_and(
            mask1,
            (torch.rand(1, batch_size, 1, device=x.device) > feature_mask_dropout_prob).to(x.dtype),
        )

        # dim: (1, batch_size, 2)
        mask = torch.cat((mask1, mask2), dim=-1)

        feature_masks = []
        for i in range(num_encoders):
            channels = self.encoder_dim[i]
            feature_mask = torch.ones(1, batch_size, channels, dtype=x.dtype, device=x.device)
            u1 = self.encoder_unmasked_dim[i]
            u2 = u1 + (channels - u1) // 2

            feature_mask[:, :, u1:u2] *= mask[..., 0:1]
            feature_mask[:, :, u2:] *= mask[..., 1:2]

            feature_masks.append(feature_mask)

        return feature_masks

    def get_chunk_info(self) -> Tuple[int, int]:
        """
        Returns chunk_size and left_context_chunks.
        """
        if not self.causal:
            return -1, -1

        if torch.jit.is_scripting() or torch.jit.is_tracing():
            assert len(self.chunk_size) == 1, self.chunk_size
            chunk_size = self.chunk_size[0]
        else:
            chunk_size = random.choice(self.chunk_size)

        if chunk_size == -1:
            left_context_chunks = -1
        else:
            if torch.jit.is_scripting() or torch.jit.is_tracing():
                assert len(self.left_context_frames) == 1, self.left_context_frames
                left_context_frames = self.left_context_frames[0]
            else:
                left_context_frames = random.choice(self.left_context_frames)
            # Note: in Python, -1 // n == -1 for n > 0
            left_context_chunks = left_context_frames // chunk_size
            if left_context_chunks == 0:
                left_context_chunks = 1

        return chunk_size, left_context_chunks

    def forward(
        self,
        x: Tensor,
        x_lens: Tensor,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
          x:
            The input tensor. Its shape is (seq_len, batch_size, feature_dim).
          x_lens:
            A tensor of shape (batch_size,) containing the number of frames in
            `x` before padding.
          src_key_padding_mask:
            The mask for padding, of shape (batch_size, seq_len); True means
            masked position. May be None.
        Returns:
          Return a tuple containing 2 tensors:
            - embeddings: its shape is (output_seq_len, batch_size, max(encoder_dim))
            - lengths, a tensor of shape (batch_size,) containing the number
              of frames in `embeddings` before padding.
        """
        outputs = []
        if torch.jit.is_scripting() or torch.jit.is_tracing():
            feature_masks = [1.0] * len(self.encoder_dim)
        else:
            feature_masks = self.get_feature_masks(x)

        chunk_size, left_context_chunks = self.get_chunk_info()

        if torch.jit.is_scripting() or torch.jit.is_tracing():
            # Not support exporting a model for simulating streaming decoding
            attn_mask = None
        else:
            attn_mask = self._get_attn_mask(x, chunk_size, left_context_chunks)

        for i, module in enumerate(self.encoders):
            x = convert_num_channels(x, self.encoder_dim[i])

            x = module(
                x,
                chunk_size=chunk_size,
                feature_mask=feature_masks[i],
                src_key_padding_mask=src_key_padding_mask,
                attn_mask=attn_mask,
            )
            outputs.append(x)

        # if the last output has the largest dimension, x will be unchanged,
        # it will be the same as outputs[-1].  Otherwise it will be concatenated
        # from different pieces of 'outputs', taking each dimension from the
        # most recent output that has it present.
        x = self._get_full_dim_output(outputs)
        # x = self.downsample_output(x)
        # class Downsample has this rounding behavior..
        # assert self.output_downsampling_factor == 2, self.output_downsampling_factor
        # if torch.jit.is_scripting() or torch.jit.is_tracing():
        #     lengths = (x_lens + 1) // 2
        # else:
        #     with warnings.catch_warnings():
        #         warnings.simplefilter("ignore")
        #         lengths = (x_lens + 1) // 2

        # return x, lengths
        return x, x_lens

    def _get_attn_mask(self, x: Tensor, chunk_size: int, left_context_chunks: int) -> Optional[Tensor]:
        """
        Return None if chunk_size == -1, else return attention mask of shape
          (seq_len, seq_len), interpreted as (tgt_seq_len, src_seq_len).  True
           means a masked position.
        Args:
           x: embeddings after self.encoder_embed(), of shape (seq_len, batch_size, embed_dim).
          chunk_size: chunk size, must divide
        """
        if chunk_size <= 0:
            return None
        assert all(chunk_size % d == 0 for d in self.downsampling_factor)
        if left_context_chunks >= 0:
            num_encoders = len(self.encoder_dim)
            assert all(chunk_size * left_context_chunks >= (self.cnn_module_kernel[i] // 2) * self.downsampling_factor[i] for i in range(num_encoders))
        else:
            left_context_chunks = 1000000

        seq_len = x.shape[0]

        # t is frame index, shape (seq_len,)
        t = torch.arange(seq_len, dtype=torch.int32, device=x.device)
        # c is chunk index for each frame, shape (seq_len,)
        if torch.jit.is_scripting() or torch.jit.is_tracing():
            c = t // chunk_size
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                c = t // chunk_size
        src_c = c
        tgt_c = c.unsqueeze(-1)

        attn_mask = torch.logical_or(src_c > tgt_c, src_c < tgt_c - left_context_chunks)
        if __name__ == "__main__":
            logging.info(f"attn_mask = {attn_mask}")
        return attn_mask

    def _get_full_dim_output(self, outputs: List[Tensor]) -> Tensor:
        num_encoders = len(self.encoder_dim)
        assert len(outputs) == num_encoders
        output_dim = max(self.encoder_dim)
        output_pieces = [outputs[-1]]
        cur_dim = self.encoder_dim[-1]
        for i in range(num_encoders - 2, -1, -1):
            d = self.encoder_dim[i]
            if d > cur_dim:
                this_output = outputs[i]
                output_pieces.append(this_output[..., cur_dim:d])
                cur_dim = d
        assert cur_dim == output_dim
        return torch.cat(output_pieces, dim=-1)

    def streaming_forward(
        self,
        x: Tensor,
        x_lens: Tensor,
        states: List[Tensor],
        src_key_padding_mask: Tensor,
    ) -> Tuple[Tensor, Tensor, List[Tensor]]:
        """
        Args:
          x:
            The input tensor. Its shape is (seq_len, batch_size, feature_dim).
          x_lens:
            A tensor of shape (batch_size,) containing the number of frames in
            `x` before padding.
          states: list of cached tensors of all encoder layers. For layer-i,
            states[i*6:(i+1)*6] is (cached_key, cached_nonlin_attn, cached_val1, cached_val2,
            cached_conv1, cached_conv2).
          src_key_padding_mask:
            The mask for padding, of shape (batch_size, seq_len); True means
            masked position. May be None.
        Returns:
          Return a tuple containing 2 tensors:
            - embeddings: its shape is (output_seq_len, batch_size, max(encoder_dim))
            - lengths, a tensor of shape (batch_size,) containing the number
              of frames in `embeddings` before padding.
            - updated states
        """
        outputs = []
        new_states = []
        layer_offset = 0

        for i, module in enumerate(self.encoders):
            num_layers = module.num_layers
            ds = self.downsampling_factor[i]
            x = convert_num_channels(x, self.encoder_dim[i])

            x, new_layer_states = module.streaming_forward(
                x,
                states=states[layer_offset * 6 : (layer_offset + num_layers) * 6],
                left_context_len=self.left_context_frames[0] // ds,
                src_key_padding_mask=src_key_padding_mask[..., ::ds],
            )
            layer_offset += num_layers
            outputs.append(x)
            new_states += new_layer_states

        # if the last output has the largest dimension, x will be unchanged,
        # it will be the same as outputs[-1].  Otherwise it will be concatenated
        # from different pieces of 'outputs', taking each dimension from the
        # most recent output that has it present.
        x = self._get_full_dim_output(outputs)
        x = self.downsample_output(x)
        # class Downsample has this rounding behavior..
        assert self.output_downsampling_factor == 2
        if torch.jit.is_scripting() or torch.jit.is_tracing():
            lengths = (x_lens + 1) // 2
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                lengths = (x_lens + 1) // 2

        return x, lengths, new_states

    @torch.jit.export
    def get_init_states(
        self,
        batch_size: int = 1,
        device: torch.device = torch.device("cpu"),
    ) -> List[Tensor]:
        """Get initial states.

        A list of cached tensors of all encoder layers. For layer-i, states[i*6:(i+1)*6]
        is (cached_key, cached_nonlin_attn, cached_val1, cached_val2, cached_conv1, cached_conv2).
        """
        states = []
        for i, module in enumerate(self.encoders):
            num_layers = module.num_layers
            embed_dim = self.encoder_dim[i]
            ds = self.downsampling_factor[i]
            num_heads = self.num_heads[i]
            key_dim = self.query_head_dim[i] * num_heads
            value_dim = self.value_head_dim[i] * num_heads
            downsample_left = self.left_context_frames[0] // ds
            nonlin_attn_head_dim = 3 * embed_dim // 4
            conv_left_pad = self.cnn_module_kernel[i] // 2
            for layer in range(num_layers):
                cached_key = torch.zeros(downsample_left, batch_size, key_dim).to(device)
                cached_nonlin_attn = torch.zeros(1, batch_size, downsample_left, nonlin_attn_head_dim).to(device)
                cached_val1 = torch.zeros(downsample_left, batch_size, value_dim).to(device)
                cached_val2 = torch.zeros(downsample_left, batch_size, value_dim).to(device)
                cached_conv1 = torch.zeros(batch_size, embed_dim, conv_left_pad).to(device)
                cached_conv2 = torch.zeros(batch_size, embed_dim, conv_left_pad).to(device)
                states += [
                    cached_key,
                    cached_nonlin_attn,
                    cached_val1,
                    cached_val2,
                    cached_conv1,
                    cached_conv2,
                ]

        return states
