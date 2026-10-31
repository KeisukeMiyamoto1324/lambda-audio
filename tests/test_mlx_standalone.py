import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mlx.core")

import mlx.nn as nn  # noqa: E402
from tokenizers import Tokenizer, models  # noqa: E402

from mlx.config import InferenceConfig, Lfm2Config, ZipformerConfig  # noqa: E402
from mlx.inference.audio import decode_audio_batch  # noqa: E402
from mlx.inference.checkpoint import save_checkpoint  # noqa: E402
from mlx.inference.preprocessing import FeatureExtractor  # noqa: E402
from mlx.model import LambdaAudioModel  # noqa: E402


def test_standalone_inference_without_project_or_training_dependencies(tmp_path: Path) -> None:
    import wave

    # ---------------------------------------------------------
    # Copy only MLX and run the actual CLI with a tiny native checkpoint.
    # Reject training imports even when they are installed on this Mac.
    # ---------------------------------------------------------
    shutil.copytree(Path(__file__).resolve().parents[1] / "mlx", tmp_path / "mlx", ignore=shutil.ignore_patterns("__pycache__"))
    encoder = ZipformerConfig(
        conv_feature_layers=((8, 3, 2),),
        encoder_dim=(8,),
        downsampling_factor=(1,),
        num_encoder_layers=(1,),
        query_head_dim=2,
        pos_head_dim=2,
        value_head_dim=2,
        pos_dim=4,
        num_heads=(2,),
        feedforward_dim=(16,),
        cnn_module_kernel=(3,),
    )
    decoder = Lfm2Config(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        layer_types=["conv", "full_attention"],
    )
    model = LambdaAudioModel(encoder, decoder)
    nn.quantize(model.decoder, group_size=64, bits=4, class_predicate=lambda _, module: isinstance(module, nn.Linear))
    save_checkpoint(model, tmp_path / "model.safetensors")
    (tmp_path / "encoder.json").write_text(encoder.model_dump_json())
    (tmp_path / "decoder.json").write_text(decoder.model_dump_json())
    features = {"feature_extractor_type": "Wav2Vec2FeatureExtractor", "padding_side": "right", "sampling_rate": 16000, "do_normalize": False, "padding_value": 0.0}
    (tmp_path / "preprocessor_config.json").write_text(json.dumps(features))
    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0, "[BOS]": 1, "[EOS]": 2, "音声": 3}, unk_token="[UNK]"))
    tokenizer.save(str(tmp_path / "tokenizer.json"))
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"bos_token": "[BOS]", "eos_token": "[EOS]", "clean_up_tokenization_spaces": False}))
    (tmp_path / ".env").write_text("HF_TOKEN=test-token\n")
    with wave.open(str(tmp_path / "audio.wav"), "wb") as output:
        output.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        output.writeframes((np.sin(np.arange(160) / 8) * 1000).astype(np.int16).tobytes())
    code = """
import importlib.abc
import pathlib
import runpy
import sys

class BlockTrainingImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path: object = None, target: object = None) -> None:
        if fullname.split(".")[0] in {"src", "torch", "transformers", "triton", "cut_cross_entropy"}:
            raise AssertionError(f"Forbidden dependency: {fullname}")

sys.meta_path.insert(0, BlockTrainingImports())
import huggingface_hub

def download(repo_id: str, filename: str, *, token: str) -> str:
    assert token == "test-token"
    if filename == "config.json":
        filename = "encoder.json" if "zipformer" in repo_id else "decoder.json"
    return str(pathlib.Path(filename).resolve())

huggingface_hub.hf_hub_download = download
sys.argv = ["benchmark", "--checkpoint", "model.safetensors", "--input", "audio.wav", "--max-new-tokens", "2", "--no-show-table"]
runpy.run_module("mlx.benchmark", run_name="__main__")
"""
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    result = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Elapsed time:" in result.stdout
    assert "Model name: lambda-audio-mlx" in result.stdout


@pytest.mark.parametrize("normalize", [False, True])
def test_numpy_features_preserve_lengths_and_padding(normalize: bool) -> None:
    extractor = FeatureExtractor(
        {
            "feature_extractor_type": "Wav2Vec2FeatureExtractor",
            "padding_side": "right",
            "sampling_rate": 16000,
            "do_normalize": normalize,
            "padding_value": 0.0,
        }
    )
    arrays = [np.array([1, 2, 3], dtype=np.float32), np.array([5, 7], dtype=np.float32)]
    batch = extractor(arrays, sampling_rate=16000, padding=True, return_attention_mask=True, return_tensors="np")
    np.testing.assert_array_equal(batch["attention_mask"], [[1, 1, 1], [1, 1, 0]])
    for index, array in enumerate(arrays):
        expected = (array - array.mean()) / np.sqrt(array.var() + 1e-7) if normalize else array
        np.testing.assert_allclose(batch["input_values"][index, : array.size], expected)
    assert batch["input_values"][1, 2] == 0
    with pytest.raises(ValueError, match="sampling rate"):
        extractor(arrays, sampling_rate=8000, padding=True, return_attention_mask=True, return_tensors="np")


def test_standalone_audio_rejects_empty_batches() -> None:
    with pytest.raises(ValueError, match="at least one"):
        decode_audio_batch([], InferenceConfig())
