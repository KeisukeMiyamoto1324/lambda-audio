from transformers import PreTrainedTokenizerBase, SequenceFeatureExtractor

from src.data.types import AudioTextBatch, TranscribedAudioExample


class AudioTextCollator:
    def __init__(
        self,
        feature_extractor: SequenceFeatureExtractor,
        tokenizer: PreTrainedTokenizerBase,
        sampling_rate: int,
    ) -> None:
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.sampling_rate = sampling_rate

        if tokenizer.bos_token_id is None or tokenizer.eos_token_id is None:
            raise ValueError("The tokenizer must define BOS and EOS tokens.")

    def __call__(
        self,
        examples: list[TranscribedAudioExample],
    ) -> AudioTextBatch:
        # ---------------------------------------------------------
        # Pad raw audio and convert its attention mask to padding mask.
        # ---------------------------------------------------------
        audio_arrays = [example["audio_array"] for example in examples]
        audio_batch = self.feature_extractor(
            audio_arrays,
            sampling_rate=self.sampling_rate,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        # ---------------------------------------------------------
        # Add explicit sequence boundaries and ignore padded labels.
        # ---------------------------------------------------------
        texts = [example["transcription"] for example in examples]
        tokenized = self.tokenizer(
            texts,
            add_special_tokens=False,
        )
        sequences = [[self.tokenizer.bos_token_id, *token_ids, self.tokenizer.eos_token_id] for token_ids in tokenized["input_ids"]]
        text_batch = self.tokenizer.pad(
            {"input_ids": sequences},
            padding=True,
            return_tensors="pt",
        )
        labels = text_batch["input_ids"].clone()
        labels.masked_fill_(text_batch["attention_mask"].eq(0), -100)

        audio_valid_samples = sum(example["audio_array"].size for example in examples)
        audio_padded_samples = audio_batch["input_values"].numel()

        return {
            "input_values": audio_batch["input_values"],
            "audio_padding_mask": audio_batch["attention_mask"].eq(0),
            "input_ids": text_batch["input_ids"],
            "attention_mask": text_batch["attention_mask"],
            "labels": labels,
            "audio_valid_samples": audio_valid_samples,
            "audio_padded_samples": audio_padded_samples,
        }
