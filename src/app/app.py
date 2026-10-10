import argparse
from pathlib import Path

import gradio as gr
from pydantic import FilePath, PositiveFloat, PositiveInt

from src.environment import load_hf_token
from src.modeling.checkpoint import load_model_checkpoint
from src.modeling.config import ModelConfig
from src.modeling.loading import load_model_components
from src.shared.device import select_device
from src.shared.transcription import transcribe


class AppConfig(ModelConfig):
    checkpoint_path: FilePath
    max_new_tokens: PositiveInt = 256
    max_audio_seconds: PositiveFloat = 30.0


def create_app(config: AppConfig) -> gr.Blocks:
    device = select_device()
    components = load_model_components(config, load_hf_token())
    load_model_checkpoint(components.model, config.checkpoint_path)
    components.model.to(device)
    components.model.eval()

    def transcribe_recording(audio_path: str) -> str:
        result = transcribe(
            components.model,
            components.feature_extractor,
            components.tokenizer,
            config,
            device,
            [Path(audio_path).read_bytes()],
        )[0]
        return result.text

    # ---------------------------------------------------------
    # Transcribe immediately after microphone recording stops.
    # ---------------------------------------------------------
    with gr.Blocks(title="LambdaAudio") as app:
        gr.Markdown("# LambdaAudio\n最大30秒まで録音できます。")
        audio = gr.Audio(
            sources="microphone",
            type="filepath",
            format="wav",
            label="録音",
        )
        output = gr.Textbox(label="文字起こし", interactive=False)
        audio.stop_recording(
            fn=transcribe_recording,
            inputs=audio,
            outputs=output,
        )
    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    arguments = parser.parse_args()
    config = AppConfig(checkpoint_path=arguments.checkpoint)
    create_app(config).launch(inbrowser=True)


if __name__ == "__main__":
    main()
