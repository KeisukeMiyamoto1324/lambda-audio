import argparse
from pathlib import Path

from pydantic import ValidationError

from src.stage2.config import Stage2Config


def parse_arguments() -> Stage2Config:
    defaults = Stage2Config.model_construct()
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=defaults.output_dir)
    parser.add_argument(
        "--metadata-index-path",
        type=Path,
        default=defaults.metadata_index_path,
    )
    parser.add_argument(
        "--shard-cache-dir",
        type=Path,
        default=defaults.shard_cache_dir,
    )
    parser.add_argument(
        "--bucket-buffer-size",
        type=int,
        default=defaults.bucket_buffer_size,
    )
    parser.add_argument("--num-workers", type=int, default=defaults.num_workers)
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=defaults.prefetch_factor,
    )
    parser.add_argument(
        "--train-max-batch-padded-audio-seconds",
        type=float,
        default=defaults.train_max_batch_padded_audio_seconds,
    )
    parser.add_argument(
        "--train-max-batch-attention-cost",
        type=int,
        default=defaults.train_max_batch_attention_cost,
    )
    parser.add_argument(
        "--validation-max-batch-padded-audio-seconds",
        type=float,
        default=defaults.validation_max_batch_padded_audio_seconds,
    )
    parser.add_argument(
        "--validation-max-batch-attention-cost",
        type=int,
        default=defaults.validation_max_batch_attention_cost,
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=defaults.gradient_accumulation_steps,
    )
    parser.add_argument("--max-steps", type=int, default=defaults.max_steps)
    parser.add_argument("--warmup-steps", type=int, default=defaults.warmup_steps)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=defaults.early_stopping_patience,
    )
    parser.add_argument(
        "--initialization-checkpoint",
        type=Path,
        default=defaults.initialization_checkpoint,
    )
    parser.add_argument(
        "--adapter-learning-rate",
        type=float,
        default=defaults.adapter_learning_rate,
    )
    parser.add_argument(
        "--backbone-learning-rate",
        type=float,
        default=defaults.backbone_learning_rate,
    )
    try:
        return Stage2Config.model_validate(vars(parser.parse_args()))
    except ValidationError as error:
        parser.error(str(error))
