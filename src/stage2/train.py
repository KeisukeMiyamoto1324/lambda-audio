import argparse
from pathlib import Path
from typing import cast

import lightning as L
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from transformers import (
    AutoFeatureExtractor,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    SequenceFeatureExtractor,
)

from src.shared.zipformer import ZipformerModel
from src.shared.environment import configure_cuda_allocator, load_hf_token
from src.shared.lambda_audio import LambdaAudioModel
from src.shared.lfm2_decoder import AudioLfm2ForCausalLM
from src.shared.performance import TrainingPerformanceMonitor, TrainingProgressBar
from src.shared.training import LambdaAudioDataModule
from src.stage2.config import Stage2Config
from src.stage2.training import Stage2LightningModule


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
        default=12.0,
    )
    parser.add_argument(
        "--validation-max-batch-attention-cost",
        type=int,
        default=12_000_000,
    )
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=204800)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--adapter-learning-rate", type=float, default=1e-5)
    parser.add_argument("--backbone-learning-rate", type=float, default=1e-6)
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


def main() -> None:
    config = parse_arguments()
    configure_cuda_allocator()
    if not torch.cuda.is_available():
        raise RuntimeError("Training requires an NVIDIA CUDA GPU.")

    # ---------------------------------------------------------
    # Configure CUDA and load pretrained audio and text components.
    # ---------------------------------------------------------
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    L.seed_everything(config.seed, workers=True)
    hf_token = load_hf_token()

    feature_extractor = cast(
        SequenceFeatureExtractor,
        AutoFeatureExtractor.from_pretrained(config.encoder_model_id, token=hf_token),
    )
    tokenizer = cast(
        PreTrainedTokenizerBase,
        AutoTokenizer.from_pretrained(config.decoder_model_id, token=hf_token),
    )
    encoder = ZipformerModel.from_pretrained(config.encoder_model_id, token=hf_token)
    decoder = AudioLfm2ForCausalLM.from_pretrained(
        config.decoder_model_id,
        attn_implementation="sdpa",
        dtype=torch.float32,
        token=hf_token,
    )
    model = LambdaAudioModel(encoder, decoder, encoder_full_precision=True)
    model.set_backbones_trainable(True)
    encoder.freeze_feature_extractor()
    model.decoder.config.use_cache = False
    model.decoder.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    lightning_model = Stage2LightningModule(model, config)
    checkpoint = torch.load(
        config.initialization_checkpoint,
        map_location="cpu",
        weights_only=True,
    )
    lightning_model.load_state_dict(checkpoint["state_dict"], strict=True)

    data_module = LambdaAudioDataModule(feature_extractor, tokenizer, config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.output_dir,
        filename="model",
        monitor="validation_loss",
        mode="min",
        every_n_train_steps=config.save_steps,
        save_top_k=1,
        save_weights_only=True,
        save_on_train_epoch_end=False,
        enable_version_counter=False,
    )
    early_stopping_callback = EarlyStopping(
        monitor="validation_loss",
        mode="min",
        patience=config.early_stopping_patience,
    )
    performance_monitor = TrainingPerformanceMonitor(
        config.sampling_rate,
        config.log_steps,
    )
    logger = CSVLogger(config.output_dir, name="logs")
    trainer = L.Trainer(
        accelerator="gpu",
        devices="auto",
        strategy="auto",
        precision="bf16-mixed",
        max_epochs=-1,
        max_steps=config.max_steps,
        callbacks=[
            checkpoint_callback,
            early_stopping_callback,
            performance_monitor,
            TrainingProgressBar(),
        ],
        logger=logger,
        accumulate_grad_batches=config.gradient_accumulation_steps,
        gradient_clip_val=config.max_grad_norm,
        log_every_n_steps=config.log_steps,
        val_check_interval=config.evaluation_steps
        * config.gradient_accumulation_steps,
        check_val_every_n_epoch=None,
        limit_val_batches=config.validation_batches,
        num_sanity_val_steps=0,
        use_distributed_sampler=False,
        default_root_dir=config.output_dir,
    )

    trainer.fit(lightning_model, datamodule=data_module)


if __name__ == "__main__":
    main()
