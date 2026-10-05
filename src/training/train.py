from src.environment import load_hf_token
from src.modeling.lfm2_decoder import enable_causal_convolution_kernels
from src.modeling.loading import load_model_components
from src.training.cli import parse_arguments
from src.training.data_module import LambdaAudioDataModule
from src.training.lightning import LambdaAudioLightningModule
from src.training.trainer import configure_training_environment, create_trainer


def main() -> None:
    config = parse_arguments()
    configure_training_environment(config.seed)
    token = load_hf_token()
    enable_causal_convolution_kernels()
    components = load_model_components(config, token)
    model = components.model
    model.decoder.config.use_cache = False

    lightning_model = LambdaAudioLightningModule(model, config)
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
