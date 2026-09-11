import copy
import random
from typing import Optional, Tuple

import torch
from torch import Tensor, nn

from .attention import RelPositionMultiheadAttentionWeights, SelfAttention
from .modules import ConvolutionModule, FeedforwardModule, NonlinAttention
from .scaling import Balancer, BiasNorm, FloatLike, ScheduledFloat, Whiten, limit_param_value
from .schedules import _whitening_schedule


class Zipformer2EncoderLayer(nn.Module):
    """
    Args:
        embed_dim: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        feedforward_dim: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        cnn_module_kernel (int): Kernel size of convolution module.

    Examples::
        >>> encoder_layer = Zipformer2EncoderLayer(embed_dim=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> pos_emb = torch.rand(32, 19, 512)
        >>> out = encoder_layer(src, pos_emb)
    """

    def __init__(
        self,
        embed_dim: int,
        pos_dim: int,
        num_heads: int,
        query_head_dim: int,
        pos_head_dim: int,
        value_head_dim: int,
        feedforward_dim: int,
        dropout: FloatLike = 0.1,
        cnn_module_kernel: int = 31,
        causal: bool = False,
        attention_skip_rate: FloatLike = ScheduledFloat((0.0, 0.2), (4000.0, 0.05), (16000, 0.0), default=0),
        conv_skip_rate: FloatLike = ScheduledFloat((0.0, 0.2), (4000.0, 0.05), (16000, 0.0), default=0),
        const_attention_rate: FloatLike = ScheduledFloat((0.0, 0.25), (4000.0, 0.025), default=0),
        ff2_skip_rate: FloatLike = ScheduledFloat((0.0, 0.1), (4000.0, 0.01), (50000.0, 0.0)),
        ff3_skip_rate: FloatLike = ScheduledFloat((0.0, 0.1), (4000.0, 0.01), (50000.0, 0.0)),
        bypass_skip_rate: FloatLike = ScheduledFloat((0.0, 0.5), (4000.0, 0.02), default=0),
    ) -> None:
        super(Zipformer2EncoderLayer, self).__init__()
        self.embed_dim = embed_dim

        # self.bypass implements layer skipping as well as bypass; see its default values.
        self.bypass = BypassModule(embed_dim, skip_rate=bypass_skip_rate, straight_through_rate=0)
        # bypass_mid is bypass used in the middle of the layer.
        self.bypass_mid = BypassModule(embed_dim, straight_through_rate=0)

        # skip probability for dynamic modules (meaning: anything but feedforward).
        self.attention_skip_rate = copy.deepcopy(attention_skip_rate)
        # an additional skip probability that applies to ConvModule to stop it from
        # contributing too much early on.
        self.conv_skip_rate = copy.deepcopy(conv_skip_rate)

        # ff2_skip_rate is to prevent the ff2 module from having output that's too big
        # compared to its residual.
        self.ff2_skip_rate = copy.deepcopy(ff2_skip_rate)
        self.ff3_skip_rate = copy.deepcopy(ff3_skip_rate)

        self.const_attention_rate = copy.deepcopy(const_attention_rate)

        self.self_attn_weights = RelPositionMultiheadAttentionWeights(
            embed_dim,
            pos_dim=pos_dim,
            num_heads=num_heads,
            query_head_dim=query_head_dim,
            pos_head_dim=pos_head_dim,
            dropout=0.0,
        )

        self.self_attn1 = SelfAttention(embed_dim, num_heads, value_head_dim)

        self.self_attn2 = SelfAttention(embed_dim, num_heads, value_head_dim)

        self.feed_forward1 = FeedforwardModule(embed_dim, (feedforward_dim * 3) // 4, dropout)

        self.feed_forward2 = FeedforwardModule(embed_dim, feedforward_dim, dropout)

        self.feed_forward3 = FeedforwardModule(embed_dim, (feedforward_dim * 5) // 4, dropout)

        self.nonlin_attention = NonlinAttention(embed_dim, hidden_channels=3 * embed_dim // 4)

        self.conv_module1 = ConvolutionModule(embed_dim, cnn_module_kernel, causal=causal)

        self.conv_module2 = ConvolutionModule(embed_dim, cnn_module_kernel, causal=causal)

        # TODO: remove it
        self.bypass_scale = nn.Parameter(torch.full((embed_dim,), 0.5))

        self.norm = BiasNorm(embed_dim)

        self.balancer1 = Balancer(
            embed_dim,
            channel_dim=-1,
            min_positive=0.45,
            max_positive=0.55,
            min_abs=0.2,
            max_abs=4.0,
        )

        # balancer for output of NonlinAttentionModule
        self.balancer_na = Balancer(
            embed_dim,
            channel_dim=-1,
            min_positive=0.3,
            max_positive=0.7,
            min_abs=ScheduledFloat((0.0, 0.004), (4000.0, 0.02)),
            prob=0.05,  # out of concern for memory usage
        )

        # balancer for output of feedforward2, prevent it from staying too
        # small.  give this a very small probability, even at the start of
        # training, it's to fix a rare problem and it's OK to fix it slowly.
        self.balancer_ff2 = Balancer(
            embed_dim,
            channel_dim=-1,
            min_positive=0.3,
            max_positive=0.7,
            min_abs=ScheduledFloat((0.0, 0.0), (4000.0, 0.1), default=0.0),
            max_abs=2.0,
            prob=0.05,
        )

        self.balancer_ff3 = Balancer(
            embed_dim,
            channel_dim=-1,
            min_positive=0.3,
            max_positive=0.7,
            min_abs=ScheduledFloat((0.0, 0.0), (4000.0, 0.2), default=0.0),
            max_abs=4.0,
            prob=0.05,
        )

        self.whiten = Whiten(
            num_groups=1,
            whitening_limit=_whitening_schedule(4.0, ratio=3.0),
            prob=(0.025, 0.25),
            grad_scale=0.01,
        )

        self.balancer2 = Balancer(
            embed_dim,
            channel_dim=-1,
            min_positive=0.45,
            max_positive=0.55,
            min_abs=0.1,
            max_abs=4.0,
        )

    def get_sequence_dropout_mask(self, x: Tensor, dropout_rate: float) -> Optional[Tensor]:
        if dropout_rate == 0.0 or not self.training or torch.jit.is_scripting() or torch.jit.is_tracing():
            return None
        batch_size = x.shape[1]
        mask = (torch.rand(batch_size, 1, device=x.device) > dropout_rate).to(x.dtype)
        return mask

    def sequence_dropout(self, x: Tensor, dropout_rate: float) -> Tensor:
        """
        Apply sequence-level dropout to x.
        x shape: (seq_len, batch_size, embed_dim)
        """
        dropout_mask = self.get_sequence_dropout_mask(x, dropout_rate)
        if dropout_mask is None:
            return x
        else:
            return x * dropout_mask

    def forward(
        self,
        src: Tensor,
        pos_emb: Tensor,
        chunk_size: int = -1,
        attn_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
            Pass the input through the encoder layer.
            Args:
                src: the sequence to the encoder (required): shape (seq_len, batch_size, embedding_dim).
             pos_emb: (1, 2*seq_len-1, pos_emb_dim) or (batch_size, 2*seq_len-1, pos_emb_dim)
             chunk_size: the number of frames per chunk, of >= 0; if -1, no chunking.
           feature_mask: something that broadcasts with src, that we'll multiply `src`
                  by at every layer: if a Tensor, likely of shape (seq_len, batch_size, embedding_dim)
             attn_mask: the attention mask, of shape (batch_size, seq_len, seq_len) or (seq_len, seq_len),
                    interpreted as (batch_size, tgt_seq_len, src_seq_len) or (tgt_seq_len, src_seq_len).
                   True means masked position. May be None.
        src_key_padding_mask:  the mask for padding, of shape (batch_size, seq_len); True means
                 masked position.  May be None.

            Returns:
               A tensor which has the same shape as src
        """
        src_orig = src

        # dropout rate for non-feedforward submodules
        if torch.jit.is_scripting() or torch.jit.is_tracing():
            attention_skip_rate = 0.0
        else:
            attention_skip_rate = float(self.attention_skip_rate) if self.training else 0.0

        # attn_weights: (num_heads, batch_size, seq_len, seq_len)
        attn_weights = self.self_attn_weights(
            src,
            pos_emb=pos_emb,
            attn_mask=attn_mask,
            key_padding_mask=src_key_padding_mask,
        )

        src = src + self.feed_forward1(src)

        self_attn_dropout_mask = self.get_sequence_dropout_mask(src, attention_skip_rate)

        selected_attn_weights = attn_weights[0:1]
        if torch.jit.is_scripting() or torch.jit.is_tracing():
            pass
        elif not self.training and random.random() < float(self.const_attention_rate):
            # Make attention weights constant.  The intention is to
            # encourage these modules to do something similar to an
            # averaging-over-time operation.
            # only need the mask, can just use the 1st one and expand later
            selected_attn_weights = selected_attn_weights[0:1]
            selected_attn_weights = (selected_attn_weights > 0.0).to(selected_attn_weights.dtype)
            selected_attn_weights = selected_attn_weights * (1.0 / selected_attn_weights.sum(dim=-1, keepdim=True))

        na = self.balancer_na(self.nonlin_attention(src, selected_attn_weights))

        src = src + (na if self_attn_dropout_mask is None else na * self_attn_dropout_mask)

        self_attn = self.self_attn1(src, attn_weights)

        src = src + (self_attn if self_attn_dropout_mask is None else self_attn * self_attn_dropout_mask)

        if torch.jit.is_scripting() or torch.jit.is_tracing():
            conv_skip_rate = 0.0
        else:
            conv_skip_rate = float(self.conv_skip_rate) if self.training else 0.0
        src = src + self.sequence_dropout(
            self.conv_module1(src, chunk_size=chunk_size, src_key_padding_mask=src_key_padding_mask),
            conv_skip_rate,
        )

        if torch.jit.is_scripting() or torch.jit.is_tracing():
            ff2_skip_rate = 0.0
        else:
            ff2_skip_rate = float(self.ff2_skip_rate) if self.training else 0.0
        src = src + self.sequence_dropout(self.balancer_ff2(self.feed_forward2(src)), ff2_skip_rate)

        # bypass in the middle of the layer.
        src = self.bypass_mid(src_orig, src)

        self_attn = self.self_attn2(src, attn_weights)

        src = src + (self_attn if self_attn_dropout_mask is None else self_attn * self_attn_dropout_mask)

        if torch.jit.is_scripting() or torch.jit.is_tracing():
            conv_skip_rate = 0.0
        else:
            conv_skip_rate = float(self.conv_skip_rate) if self.training else 0.0
        src = src + self.sequence_dropout(
            self.conv_module2(src, chunk_size=chunk_size, src_key_padding_mask=src_key_padding_mask),
            conv_skip_rate,
        )

        if torch.jit.is_scripting() or torch.jit.is_tracing():
            ff3_skip_rate = 0.0
        else:
            ff3_skip_rate = float(self.ff3_skip_rate) if self.training else 0.0
        src = src + self.sequence_dropout(self.balancer_ff3(self.feed_forward3(src)), ff3_skip_rate)

        src = self.balancer1(src)
        src = self.norm(src)

        src = self.bypass(src_orig, src)

        src = self.balancer2(src)
        src = self.whiten(src)

        return src

    def streaming_forward(
        self,
        src: Tensor,
        pos_emb: Tensor,
        cached_key: Tensor,
        cached_nonlin_attn: Tensor,
        cached_val1: Tensor,
        cached_val2: Tensor,
        cached_conv1: Tensor,
        cached_conv2: Tensor,
        left_context_len: int,
        src_key_padding_mask: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Pass the input through the encoder layer in streaming forward mode.

        Args:
            src: the sequence to the encoder (required): shape (seq_len, batch_size, embedding_dim).
            pos_emb: (1, left_context_len+2*seq_len-1, pos_emb_dim) or
              (batch_size, left_context_len+2*seq_len-1, pos_emb_dim)
            cached_key: cached attention key tensor of left context,
              of shape (left_context_len, batch_size, key_dim)
            cached_nonlin_attn: left context for nonlin_attention module, a Tensor of shape
              (num_heads, batch_size, left_context_len, head_dim)
            cached_val1: cached left context for the first attention module,
              of shape (left_context_len, batch_size, value_dim)
            cached_val2: cached left context for the second attention module,
              of shape (left_context_len, batch_size, value_dim)
            cached_conv1: cached left context for the first convolution module,
              of shape (batch_size, channels, left_pad)
            cached_conv2: cached left context for the second convolution module,
              of shape (batch_size, channels, left_pad)
            left_context_len: number of left context frames.
            src_key_padding_mask:  the mask for padding, of shape
              (batch_size, left_context_len + seq_len); True means masked position.
              May be None.

        Returns:
            - x, with the same shape as src
            - updated cached_key
            - updated cached_nonlin_attn
            - updated cached_val1
            - updated cached_val2
            - updated cached_conv1
            - updated cached_conv2
        """
        src_orig = src

        # attn_weights: (num_heads, batch_size, seq_len, seq_len)
        attn_weights, cached_key = self.self_attn_weights.streaming_forward(
            src,
            pos_emb=pos_emb,
            cached_key=cached_key,
            left_context_len=left_context_len,
            key_padding_mask=src_key_padding_mask,
        )

        src = src + self.feed_forward1(src)

        na, cached_nonlin_attn = self.nonlin_attention.streaming_forward(
            src,
            attn_weights[0:1],
            cached_x=cached_nonlin_attn,
            left_context_len=left_context_len,
        )
        src = src + na

        self_attn, cached_val1 = self.self_attn1.streaming_forward(
            src,
            attn_weights=attn_weights,
            cached_val=cached_val1,
            left_context_len=left_context_len,
        )
        src = src + self_attn

        src_conv, cached_conv1 = self.conv_module1.streaming_forward(
            src,
            cache=cached_conv1,
            src_key_padding_mask=src_key_padding_mask[:, left_context_len:],
        )
        src = src + src_conv

        src = src + self.feed_forward2(src)

        # bypass in the middle of the layer.
        src = self.bypass_mid(src_orig, src)

        self_attn, cached_val2 = self.self_attn2.streaming_forward(
            src,
            attn_weights=attn_weights,
            cached_val=cached_val2,
            left_context_len=left_context_len,
        )
        src = src + self_attn

        src_conv, cached_conv2 = self.conv_module2.streaming_forward(
            src,
            cache=cached_conv2,
            src_key_padding_mask=src_key_padding_mask[:, left_context_len:],
        )
        src = src + src_conv

        src = src + self.feed_forward3(src)

        src = self.norm(src)

        src = self.bypass(src_orig, src)

        return (
            src,
            cached_key,
            cached_nonlin_attn,
            cached_val1,
            cached_val2,
            cached_conv1,
            cached_conv2,
        )


class BypassModule(nn.Module):
    """
    An nn.Module that implements a learnable bypass scale, and also randomized per-sequence
    layer-skipping.  The bypass is limited during early stages of training to be close to
    "straight-through", i.e. to not do the bypass operation much initially, in order to
    force all the modules to learn something.
    """

    def __init__(
        self,
        embed_dim: int,
        skip_rate: FloatLike = 0.0,
        straight_through_rate: FloatLike = 0.0,
        scale_min: FloatLike = ScheduledFloat((0.0, 0.9), (20000.0, 0.2), default=0),
        scale_max: FloatLike = 1.0,
    ) -> None:
        super().__init__()
        self.bypass_scale = nn.Parameter(torch.full((embed_dim,), 0.5))
        self.skip_rate = copy.deepcopy(skip_rate)
        self.straight_through_rate = copy.deepcopy(straight_through_rate)
        self.scale_min = copy.deepcopy(scale_min)
        self.scale_max = copy.deepcopy(scale_max)

    def _get_bypass_scale(self, batch_size: int) -> Tensor:
        # returns bypass-scale of shape (num_channels,),
        # or (batch_size, num_channels,).  This is actually the
        # scale on the non-residual term, so 0 correponds to bypassing
        # this module.
        if torch.jit.is_scripting() or torch.jit.is_tracing() or not self.training:
            return self.bypass_scale
        else:
            ans = limit_param_value(self.bypass_scale, min=float(self.scale_min), max=float(self.scale_max))
            skip_rate = float(self.skip_rate)
            if skip_rate != 0.0:
                mask = torch.rand((batch_size, 1), device=ans.device) > skip_rate
                ans = ans * mask
                # now ans is of shape (batch_size, num_channels), and is zero for sequences
                # on which we have randomly chosen to do layer-skipping.
            straight_through_rate = float(self.straight_through_rate)
            if straight_through_rate != 0.0:
                mask = torch.rand((batch_size, 1), device=ans.device) < straight_through_rate
                ans = torch.maximum(ans, mask.to(ans.dtype))
            return ans

    def forward(self, src_orig: Tensor, src: Tensor) -> Tensor:
        """
        Args: src_orig and src are both of shape (seq_len, batch_size, num_channels)
        Returns: something with the same shape as src and src_orig
        """
        bypass_scale = self._get_bypass_scale(src.shape[1])
        return src_orig + (src - src_orig) * bypass_scale
