import argparse
from pathlib import Path
from typing import cast

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from transformers import (
    AutoFeatureExtractor,
    AutoModel,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    SequenceFeatureExtractor,
)

from src.collator import AudioTextCollator
from src.config import TrainingConfig
from src.dataset import load_streaming_datasets
from src.lambda_audio import LambdaAudioModel
from src.lfm2_decoder import AudioLfm2ForCausalLM
from src.training import create_dataloader, create_optimizer, create_scheduler, train


def parse_arguments() -> TrainingConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, choices=(1, 2), required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--initialization-checkpoint", type=Path)
    parser.add_argument("--dataset-config", default="tiny")
    parser.add_argument("--train-batch-size", type=int, default=2)
    parser.add_argument("--validation-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=10_000)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--dataloader-workers", type=int, default=4)
    parser.add_argument("--mixed-precision", choices=("bf16", "fp16"), default="bf16")
    arguments = parser.parse_args()

    if arguments.stage == 2 and arguments.initialization_checkpoint is None:
        parser.error("Stage 2 requires --initialization-checkpoint.")
    if arguments.dataloader_workers < 1:
        parser.error("--dataloader-workers must be at least 1.")

    return TrainingConfig(
        stage=arguments.stage,
        output_dir=arguments.output_dir,
        initialization_checkpoint=arguments.initialization_checkpoint,
        dataset_config=arguments.dataset_config,
        train_batch_size=arguments.train_batch_size,
        validation_batch_size=arguments.validation_batch_size,
        gradient_accumulation_steps=arguments.gradient_accumulation_steps,
        max_steps=arguments.max_steps,
        warmup_steps=arguments.warmup_steps,
        dataloader_workers=arguments.dataloader_workers,
        mixed_precision=arguments.mixed_precision,
    )


def main() -> None:
    config = parse_arguments()
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        step_scheduler_with_optimizer=False,
    )
    if accelerator.device.type != "cuda":
        raise RuntimeError("Training requires an NVIDIA CUDA GPU.")

    # ---------------------------------------------------------
    # Configure CUDA and load pretrained audio and text components.
    # ---------------------------------------------------------
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(config.seed)

    feature_extractor = cast(
        SequenceFeatureExtractor,
        AutoFeatureExtractor.from_pretrained(config.encoder_model_id),
    )
    tokenizer = cast(
        PreTrainedTokenizerBase,
        AutoTokenizer.from_pretrained(config.decoder_model_id),
    )
    encoder = cast(
        PreTrainedModel,
        AutoModel.from_pretrained(config.encoder_model_id, trust_remote_code=True),
    )
    decoder = AudioLfm2ForCausalLM.from_pretrained(
        config.decoder_model_id,
        attn_implementation="sdpa",
        dtype=torch.float32,
    )
    model = LambdaAudioModel(encoder, decoder)

    # ---------------------------------------------------------
    # Build each DDP stage separately so all trainable weights synchronize.
    # ---------------------------------------------------------
    if config.initialization_checkpoint is not None:
        state_dict = torch.load(
            config.initialization_checkpoint,
            map_location="cpu",
            weights_only=True,
        )
        model.load_state_dict(state_dict, strict=True)

    model.set_backbones_trainable(config.stage == 2)
    model.decoder.config.use_cache = False
    if config.stage == 2:
        model.decoder.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    training_dataset, validation_dataset = load_streaming_datasets(config)
    collator = AudioTextCollator(
        feature_extractor,
        tokenizer,
        config.sampling_rate,
        config.max_text_length,
    )
    training_dataloader = create_dataloader(
        training_dataset,
        collator,
        config.train_batch_size,
        config,
    )
    validation_dataloader = create_dataloader(
        validation_dataset,
        collator,
        config.validation_batch_size,
        config,
    )
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)

    model, optimizer, training_dataloader, validation_dataloader, scheduler = (
        accelerator.prepare(
            model,
            optimizer,
            training_dataloader,
            validation_dataloader,
            scheduler,
        )
    )
    train(
        model,
        training_dataloader,
        validation_dataloader,
        optimizer,
        scheduler,
        accelerator,
        config,
    )


if __name__ == "__main__":
    main()
