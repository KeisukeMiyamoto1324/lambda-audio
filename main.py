from typing import cast

from transformers import AutoModel, PreTrainedModel

from str.lambda_audio import LambdaAudioModel
from str.lfm2_decoder import AudioLfm2ForCausalLM


ENCODER_MODEL_ID = "reazon-research/japanese-zipformer-base-k2"
DECODER_MODEL_ID = "LiquidAI/LFM2.5-230M-Base"


def main() -> None:
    # ---------------------------------------------------------
    # Load both pretrained backbones and build LambdaAudio.
    # ---------------------------------------------------------
    encoder = cast(
        PreTrainedModel,
        AutoModel.from_pretrained(ENCODER_MODEL_ID, trust_remote_code=True),
    )
    decoder = AudioLfm2ForCausalLM.from_pretrained(DECODER_MODEL_ID)
    model = LambdaAudioModel(encoder, decoder)
    model.set_backbones_trainable(False)

    print(model)


if __name__ == "__main__":
    main()
