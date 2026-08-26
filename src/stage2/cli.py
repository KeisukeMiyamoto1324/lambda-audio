import argparse
from pathlib import Path

from src.stage2.config import Stage2Config


def parse_arguments() -> Stage2Config:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--initialization-checkpoint",
        type=Path,
        default=Path("outputs/lambda-audio-stage1/model.ckpt"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/stage2"))
    parser.add_argument(
        "--metadata-index-path",
        type=Path,
        default=Path("outputs/datasets/reazon-speech-v2-metadata.sqlite3"),
    )
    parser.add_argument(
        "--shard-cache-dir",
        type=Path,
        default=Path("outputs/datasets/reazon-speech-v2-shards"),
    )
    parser.add_argument("--bucket-buffer-size", type=int, default=512)
    parser.add_argument(
        "--num-workers",
        type=int,
        choices=range(4, 33),
        default=16,
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        choices=range(2, 9),
        default=4,
    )
    parser.add_argument(
        "--max-batch-padded-audio-seconds",
        type=float,
        default=1600.0,
    )
    parser.add_argument(
        "--max-batch-attention-cost",
        type=int,
        default=40_000_000,
    )
    parser.add_argument(
        "--validation-max-batch-padded-audio-seconds",
        type=float,
        default=256.0,
    )
    parser.add_argument(
        "--validation-max-batch-attention-cost",
        type=int,
        default=36_000_000,
    )
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=204800)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--adapter-learning-rate", type=float, default=2e-5)
    parser.add_argument("--backbone-learning-rate", type=float, default=2e-5)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    arguments = parser.parse_args()

    if arguments.max_batch_padded_audio_seconds <= 0:
        parser.error("--max-batch-padded-audio-seconds must be positive.")
    if arguments.max_batch_attention_cost < 1:
        parser.error("--max-batch-attention-cost must be positive.")
    if arguments.validation_max_batch_padded_audio_seconds <= 0:
        parser.error(
            "--validation-max-batch-padded-audio-seconds must be positive."
        )
    if arguments.validation_max_batch_attention_cost < 1:
        parser.error("--validation-max-batch-attention-cost must be positive.")
    if arguments.bucket_buffer_size < 1:
        parser.error("--bucket-buffer-size must be at least 1.")
    return Stage2Config(
        output_dir=arguments.output_dir,
        metadata_index_path=arguments.metadata_index_path,
        shard_cache_dir=arguments.shard_cache_dir,
        bucket_buffer_size=arguments.bucket_buffer_size,
        num_workers=arguments.num_workers,
        prefetch_factor=arguments.prefetch_factor,
        initialization_checkpoint=arguments.initialization_checkpoint,
        train_max_batch_padded_audio_seconds=(
            arguments.max_batch_padded_audio_seconds
        ),
        validation_max_batch_padded_audio_seconds=(
            arguments.validation_max_batch_padded_audio_seconds
        ),
        train_max_batch_attention_cost=arguments.max_batch_attention_cost,
        validation_max_batch_attention_cost=(
            arguments.validation_max_batch_attention_cost
        ),
        gradient_accumulation_steps=arguments.gradient_accumulation_steps,
        max_steps=arguments.max_steps,
        warmup_steps=arguments.warmup_steps,
        adapter_learning_rate=arguments.adapter_learning_rate,
        backbone_learning_rate=arguments.backbone_learning_rate,
        early_stopping_patience=arguments.early_stopping_patience,
    )
