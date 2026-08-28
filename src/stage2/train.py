from src.environment import load_hf_token
from src.modeling.checkpoint import load_checkpoint_state
from src.modeling.loading import load_model_components
from src.stage2.cli import parse_arguments
from src.stage2.training import Stage2LightningModule
from src.training.data_module import LambdaAudioDataModule
from src.training.trainer import configure_training_environment, create_trainer


def main() -> None:
    config = parse_arguments()
    configure_training_environment(config.seed)
    token = load_hf_token()
    components = load_model_components(
        config,
        token,
        encoder_full_precision=True,
    )
    model = components.model
    model.encoder.requires_grad_(False)
    model.decoder.config.use_cache = False
    model.decoder.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    lightning_model = Stage2LightningModule(model, config)
    if config.resume_checkpoint is None:
        lightning_model.load_state_dict(
            load_checkpoint_state(config.initialization_checkpoint),
            strict=True,
        )
    data_module = LambdaAudioDataModule(
        components.feature_extractor,
        components.tokenizer,
        config,
        token,
    )
    create_trainer(config).fit(
        lightning_model,
        datamodule=data_module,
        ckpt_path=config.resume_checkpoint,
    )


if __name__ == "__main__":
    main()
