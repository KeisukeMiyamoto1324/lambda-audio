import torch
from transformers import PreTrainedTokenizerBase, SequenceFeatureExtractor

from src.data.types import TranscribedAudioExample


class AudioTextCollator:
    def __init__(
        self,
        feature_extractor: SequenceFeatureExtractor,
        tokenizer: PreTrainedTokenizerBase,
        ctc_tokenizer: PreTrainedTokenizerBase,
        sampling_rate: int,
        max_text_length: int,
    ) -> None:
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.ctc_tokenizer = ctc_tokenizer
        self.sampling_rate = sampling_rate
        self.max_text_length = max_text_length

        if tokenizer.bos_token_id is None or tokenizer.eos_token_id is None:
            raise ValueError("The tokenizer must define BOS and EOS tokens.")
        if ctc_tokenizer.pad_token_id is None:
            raise ValueError("The CTC tokenizer must define a blank token.")

    def __call__(
        self,
        examples: list[TranscribedAudioExample],
    ) -> dict[str, torch.Tensor | int]:
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
            truncation=True,
            max_length=self.max_text_length - 2,
        )
        sequences = [[self.tokenizer.bos_token_id, *token_ids, self.tokenizer.eos_token_id] for token_ids in tokenized["input_ids"]]
        text_batch = self.tokenizer.pad(
            {"input_ids": sequences},
            padding=True,
            return_tensors="pt",
        )
        labels = text_batch["input_ids"].clone()
        labels.masked_fill_(text_batch["attention_mask"].eq(0), -100)

        # ---------------------------------------------------------
        # Build separate labels without sequence boundary tokens for CTC.
        # ---------------------------------------------------------
        ctc_batch = self.ctc_tokenizer(
            texts,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_text_length,
            padding=True,
            return_tensors="pt",
        )
        ctc_labels = ctc_batch["input_ids"]
        ctc_label_lengths = ctc_batch["attention_mask"].sum(dim=-1)
        active_ctc_labels = ctc_labels[ctc_batch["attention_mask"].bool()]
        if active_ctc_labels.eq(self.ctc_tokenizer.pad_token_id).any():
            raise ValueError("CTC targets must not contain the blank token.")

        return {
            "input_values": audio_batch["input_values"],
            "audio_padding_mask": audio_batch["attention_mask"].eq(0),
            "input_ids": text_batch["input_ids"],
            "attention_mask": text_batch["attention_mask"],
            "labels": labels,
            "ctc_labels": ctc_labels,
            "ctc_label_lengths": ctc_label_lengths,
            "audio_valid_samples": sum(audio_array.size for audio_array in audio_arrays),
        }
