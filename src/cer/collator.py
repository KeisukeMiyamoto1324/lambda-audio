from transformers import SequenceFeatureExtractor

from src.cer.types import CerBatch
from src.data.types import TranscribedAudioExample


class CerCollator:
    def __init__(
        self,
        feature_extractor: SequenceFeatureExtractor,
        sampling_rate: int,
    ) -> None:
        self.feature_extractor = feature_extractor
        self.sampling_rate = sampling_rate

    def __call__(self, examples: list[TranscribedAudioExample]) -> CerBatch:
        audio_batch = self.feature_extractor(
            [example["audio_array"] for example in examples],
            sampling_rate=self.sampling_rate,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return {
            "input_values": audio_batch["input_values"],
            "audio_padding_mask": audio_batch["attention_mask"].eq(0),
            "references": [example["transcription"] for example in examples],
        }
