from typing import Any

import numpy as np
from numpy.typing import NDArray
from tokenizers import Tokenizer as NativeTokenizer


class FeatureExtractor:
    def __init__(self, config: dict[str, Any]) -> None:
        if config["feature_extractor_type"] != "Wav2Vec2FeatureExtractor" or config["padding_side"] != "right":
            raise ValueError("Only right-padded Wav2Vec2 audio features are supported.")
        self.sampling_rate = config["sampling_rate"]
        self.do_normalize = config["do_normalize"]
        self.padding_value = config["padding_value"]

    def __call__(self, arrays: list[NDArray[np.float32]], *, sampling_rate: int, padding: bool, return_attention_mask: bool, return_tensors: str) -> dict[str, NDArray[Any]]:
        if sampling_rate != self.sampling_rate or not padding or not return_attention_mask or return_tensors != "np":
            raise ValueError("Expected padded NumPy features at the configured sampling rate.")
        values = np.full((len(arrays), max(array.size for array in arrays)), self.padding_value, dtype=np.float32)
        mask = np.zeros(values.shape, dtype=np.int32)
        for index, array in enumerate(arrays):
            if self.do_normalize:
                array = (array - array.mean()) / np.sqrt(array.var() + 1e-7)
            values[index, : array.size] = array
            mask[index, : array.size] = 1
        return {"input_values": values, "attention_mask": mask}


class Tokenizer:
    def __init__(self, path: str, config: dict[str, Any]) -> None:
        self.backend = NativeTokenizer.from_file(path)
        self.bos_token_id = self.backend.token_to_id(config["bos_token"])
        self.eos_token_id = self.backend.token_to_id(config["eos_token"])
        if self.bos_token_id is None or self.eos_token_id is None:
            raise ValueError("The tokenizer must define BOS and EOS tokens.")
        if config["clean_up_tokenization_spaces"]:
            raise ValueError("Tokenization space cleanup is not supported.")

    def batch_decode(self, tokens: list[list[int]], skip_special_tokens: bool) -> list[str]:
        return self.backend.decode_batch(tokens, skip_special_tokens=skip_special_tokens)
