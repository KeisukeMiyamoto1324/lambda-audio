from typing import Any, cast

import torch

from src.environment import load_hf_token
from src.inference.loader import select_device
from src.modeling.checkpoint import load_model_checkpoint
from src.modeling.loading import load_model_components
from src.test.cli import parse_arguments
from src.test.dataset import load_local_test_dataset
from src.test.evaluation import evaluate_test_dataset


def main() -> None:
    config = parse_arguments()
    token = load_hf_token()
    device = select_device()
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    components = load_model_components(config, token)
    load_model_checkpoint(components.model, config.checkpoint_path)
    components.model.to(device)
    components.model.eval()

    rows = load_local_test_dataset(config.dataset_dir)
    summary = evaluate_test_dataset(
        components.model,
        components.feature_extractor,
        components.tokenizer,
        config,
        device,
        cast(Any, rows),
    )
    print(f"Samples: {summary.samples}")
    print(f"Character errors: {summary.character_errors}")
    print(f"Reference characters: {summary.reference_characters}")
    print(f"CER: {summary.cer:.6f}")
    print(f"Results: {config.output_path}")


if __name__ == "__main__":
    main()
