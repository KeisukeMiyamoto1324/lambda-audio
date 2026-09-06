from typing import TypedDict

import numpy as np
from numpy.typing import NDArray


class TranscribedAudioExample(TypedDict):
    audio_array: NDArray[np.float32]
    transcription: str
