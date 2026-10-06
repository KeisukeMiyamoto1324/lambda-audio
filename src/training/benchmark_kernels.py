import argparse
import json
import random
import statistics
import time
from pathlib import Path
from typing import Any

import torch

from src.environment import load_hf_token
from src.modeling.lfm2_decoder import enable_causal_convolution_kernels
from src.modeling.loading import load_model_components
from src.modeling.zipformer.scaling import ScheduledFloat
from src.training.config import TrainingConfig
from src.training.kernels import enable_training_kernels
from src.training.optimizer import create_optimizer_configuration
from src.training.trainer import configure_training_environment


class TrainingBenchmark:
    def __init__(self, backend: str, frozen: bool, batch_size: int, seconds: float, text_length: int, accumulation: int) -> None:
        config = TrainingConfig()
        configure_training_environment(42)
        token = load_hf_token()
        enable_causal_convolution_kernels()
        self.model = load_model_components(config, token).model.cuda().train()
        if backend == "triton":
            enable_training_kernels(self.model.encoder)
        self.model.decoder.config.use_cache = False
        self.model.set_backbones_trainable(not frozen)
        for module in self.model.encoder.modules():
            if isinstance(module, ScheduledFloat):
                module.batch_count = 1000 if frozen else 5000
        optimization = create_optimizer_configuration(self.model, config)
        self.optimizer = optimization["optimizer"]
        self.scheduler = optimization["lr_scheduler"]["scheduler"]
        samples = int(seconds * config.sampling_rate)
        lengths = torch.linspace(samples * 0.75, samples, batch_size, device="cuda").long()
        audio_mask = torch.arange(samples, device="cuda")[None, :] >= lengths[:, None]
        self.batch = {
            "input_values": torch.randn(batch_size, samples, device="cuda").masked_fill(audio_mask, 0),
            "audio_padding_mask": audio_mask,
            "input_ids": torch.randint(8, self.model.decoder.config.vocab_size, (batch_size, text_length), device="cuda"),
            "attention_mask": torch.ones(batch_size, text_length, device="cuda", dtype=torch.long),
        }
        self.batch["labels"] = self.batch["input_ids"].clone()
        self.token_count = batch_size * (text_length - 1)
        self.accumulation = accumulation
        self.loss: torch.Tensor | None = None

    def step(self) -> None:
        self.optimizer.zero_grad(set_to_none=True)
        for _ in range(self.accumulation):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = self.model(**self.batch)
                loss = output.loss / self.accumulation
            loss.backward()
            self.loss = loss.detach()
        for parameter in self.model.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(1.0 / self.token_count)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0, error_if_nonfinite=True, foreach=True)
        self.optimizer.step()
        self.scheduler.step()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("torch", "triton"), required=True)
    parser.add_argument("--frozen", action="store_true")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--audio-seconds", type=float, default=10.0)
    parser.add_argument("--text-length", type=int, default=128)
    parser.add_argument("--accumulation", type=int, default=3)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.batch_size, args.audio_seconds, args.accumulation, args.steps, args.warmup) <= 0 or args.text_length < 2:
        raise ValueError("Benchmark dimensions and counts must be positive; text length must be at least two.")
    benchmark = TrainingBenchmark(args.backend, args.frozen, args.batch_size, args.audio_seconds, args.text_length, args.accumulation)
    for _ in range(args.warmup):
        benchmark.step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    timings = []
    for index in range(args.steps):
        random.seed(100 + index)
        torch.manual_seed(100 + index)
        start = time.perf_counter()
        benchmark.step()
        torch.cuda.synchronize()
        timings.append((time.perf_counter() - start) * 1000)
    result: dict[str, Any] = {
        **{key: value for key, value in vars(args).items() if key != "output"},
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "median_step_ms": statistics.median(timings),
        "mean_step_ms": statistics.mean(timings),
        "step_times_ms": timings,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "last_loss_per_token": benchmark.loss.item() * args.accumulation / benchmark.token_count,
        "scope": "Synthetic audio/text; forward, backward, clipping, fused AdamW; excludes data loading, validation and saving.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
