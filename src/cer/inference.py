import torch
from transformers import PreTrainedTokenizerBase

from src.cer.types import CerBatch
from src.modeling.lambda_audio import LambdaAudioModel


@torch.inference_mode()
def generate_transcriptions(
    model: LambdaAudioModel,
    tokenizer: PreTrainedTokenizerBase,
    batch: CerBatch,
    max_new_tokens: int,
) -> list[str]:
    if tokenizer.bos_token_id is None or tokenizer.eos_token_id is None:
        raise ValueError("The tokenizer must define BOS and EOS tokens.")

    batch_size = batch["input_values"].shape[0]
    input_ids = torch.full(
        (batch_size, 1),
        tokenizer.bos_token_id,
        dtype=torch.long,
        device=batch["input_values"].device,
    )
    generated_ids = model.generate(
        input_values=batch["input_values"],
        audio_padding_mask=batch["audio_padding_mask"],
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        eos_token_id=tokenizer.eos_token_id,
    )
    transcription_ids = generated_ids[:, input_ids.shape[1] :]
    return [text.strip() for text in tokenizer.batch_decode(transcription_ids, skip_special_tokens=True)]
