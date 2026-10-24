from typing import TypedDict

import numpy as np
import torch
from numpy.typing import NDArray


class TranscribedAudioExample(TypedDict):
    audio_array: NDArray[np.float32]
    transcription: str


class AudioTextBatch(TypedDict):
    input_values: torch.Tensor
    audio_padding_mask: torch.BoolTensor
    input_ids: torch.LongTensor
    attention_mask: torch.Tensor
    labels: torch.LongTensor
    target_token_count: int
    audio_valid_samples: int
    audio_padded_samples: int
