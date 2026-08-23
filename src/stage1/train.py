import argparse
from pathlib import Path
from typing import cast

import lightning as L
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar
from transformers import (
    AutoFeatureExtractor,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    SequenceFeatureExtractor,
)

from src.encoders.zipformer import ZipformerModel
from src.shared.environment import load_hf_token
from src.shared.lambda_audio import LambdaAudioModel
from src.shared.lfm2_decoder import AudioLfm2ForCausalLM
from src.shared.training import LambdaAudioDataModule
from src.stage1.config import Stage1Config
from src.stage1.training import Stage1LightningModule


def parse_arguments() -> Stage1Config:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/stage1"))
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/prepared/reazonspeech-tiny"),
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--validation-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--bucket-buffer-multiplier", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=10_000)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--dataloader-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    arguments = parser.parse_args()

    if arguments.dataloader_workers < 1:
        parser.error("--dataloader-workers must be at least 1.")
    if arguments.bucket_buffer_multiplier < 1:
        parser.error("--bucket-buffer-multiplier must be at least 1.")

    return Stage1Config(
        output_dir=arguments.output_dir,
        data_dir=arguments.data_dir,
        train_batch_size=arguments.batch_size,
        validation_batch_size=arguments.validation_batch_size,
        gradient_accumulation_steps=arguments.gradient_accumulation_steps,
        bucket_buffer_multiplier=arguments.bucket_buffer_multiplier,
        max_steps=arguments.max_steps,
        warmup_steps=arguments.warmup_steps,
        dataloader_workers=arguments.dataloader_workers,
        learning_rate=arguments.learning_rate,
        early_stopping_patience=arguments.early_stopping_patience,
    )


def main() -> None:
    config = parse_arguments()
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
    model = LambdaAudioModel(encoder, decoder)
    model.set_backbones_trainable(False)
    model.decoder.config.use_cache = False

    lightning_model = Stage1LightningModule(model, config)
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
            TQDMProgressBar(),
        ],
        logger=False,
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

    # lightning_model.model = torch.compile(lightning_model.model)
    trainer.fit(lightning_model, datamodule=data_module)


if __name__ == "__main__":
    main()
