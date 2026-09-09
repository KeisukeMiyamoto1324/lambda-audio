import json
import unicodedata
from collections.abc import Iterable
from time import perf_counter
from typing import Any, cast

import numpy as np
import torch
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, NonNegativeFloat, NonNegativeInt
from rich.progress import track
from transformers import PreTrainedTokenizerBase, SequenceFeatureExtractor

from src.inference.transcription import transcribe_audio_array
from src.modeling.lambda_audio import LambdaAudioModel
from src.test.config import EvaluationConfig

TEST_DATASET_ID = "KeisukeMiyamoto/lambda-audio-courpus-1"
TEST_DATASET_SIZE = 28_653


class EvaluationSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    samples: NonNegativeInt
    character_errors: NonNegativeInt
    reference_characters: NonNegativeInt
    cer: NonNegativeFloat


def normalize_for_cer(text: str) -> str:
    return "".join(character for character in text if not character.isspace() and not unicodedata.category(character).startswith("P"))


def character_edit_distance(reference: str, prediction: str) -> int:
    previous_row = list(range(len(prediction) + 1))
    for reference_index, reference_character in enumerate(reference, start=1):
        current_row = [reference_index]
        for prediction_index, prediction_character in enumerate(prediction, start=1):
            substitution_cost = reference_character != prediction_character
            current_row.append(
                min(
                    current_row[-1] + 1,
                    previous_row[prediction_index] + 1,
                    previous_row[prediction_index - 1] + substitution_cost,
                )
            )
        previous_row = current_row
    return previous_row[-1]


def evaluate_test_dataset(
    model: LambdaAudioModel,
    feature_extractor: SequenceFeatureExtractor,
    tokenizer: PreTrainedTokenizerBase,
    config: EvaluationConfig,
    device: torch.device,
    rows: Iterable[dict[str, Any]],
) -> EvaluationSummary:
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    sample_count = 0
    total_errors = 0
    total_reference_characters = 0

    # ---------------------------------------------------------
    # Write each result immediately while accumulating corpus CER.
    # ---------------------------------------------------------
    with config.output_path.open("w", encoding="utf-8") as output_file:
        for row in track(rows, total=TEST_DATASET_SIZE, description="Testing"):
            samples = row["audio"].get_all_samples()
            audio_array = cast(
                NDArray[np.float32],
                samples.data.squeeze(0).numpy().astype(np.float32, copy=False),
            )
            inference_start = perf_counter()
            result = transcribe_audio_array(
                model,
                feature_extractor,
                tokenizer,
                config,
                device,
                audio_array,
                samples.sample_rate,
            )
            inference_seconds = perf_counter() - inference_start
            reference = cast(str, row["text"])
            normalized_reference = normalize_for_cer(reference)
            normalized_prediction = normalize_for_cer(result.text)
            errors = character_edit_distance(normalized_reference, normalized_prediction)

            record = {
                "id": row["id"],
                "reference": reference,
                "prediction": result.text,
                "normalized_reference": normalized_reference,
                "normalized_prediction": normalized_prediction,
                "character_errors": errors,
                "reference_characters": len(normalized_reference),
                "audio_seconds": result.audio_seconds,
                "inference_seconds": inference_seconds,
            }
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            output_file.flush()
            sample_count += 1
            total_errors += errors
            total_reference_characters += len(normalized_reference)

    if total_reference_characters == 0:
        raise ValueError("The test dataset contains no reference characters.")
    return EvaluationSummary(
        samples=sample_count,
        character_errors=total_errors,
        reference_characters=total_reference_characters,
        cer=total_errors / total_reference_characters,
    )
