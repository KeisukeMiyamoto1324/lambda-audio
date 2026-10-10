import ast
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.benchmark.config import BenchmarkConfig
from src.training.config import TrainingConfig


def test_training_config_validates_positive_values() -> None:
    with pytest.raises(ValidationError):
        TrainingConfig(num_workers=0)

    with pytest.raises(ValidationError):
        TrainingConfig(min_learning_rate_ratio=1.1)


def test_training_config_is_frozen() -> None:
    config = TrainingConfig()

    with pytest.raises(ValidationError):
        config.max_steps = 1  # type: ignore[misc]


@pytest.mark.parametrize("unfreeze_step", [-1, 10, 11])
def test_training_config_rejects_invalid_unfreeze_step(unfreeze_step: int) -> None:
    with pytest.raises(ValidationError):
        TrainingConfig(max_steps=10, unfreeze_step=unfreeze_step)


def test_resume_validates_checkpoint_path(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        TrainingConfig(resume_checkpoint=tmp_path / "missing.ckpt")
    checkpoint = tmp_path / "last.ckpt"
    checkpoint.touch()
    assert TrainingConfig(resume_checkpoint=checkpoint).resume_checkpoint == checkpoint


def test_inference_config_validates_paths(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "model.ckpt"
    checkpoint_path.touch()

    config = BenchmarkConfig(
        checkpoint_path=checkpoint_path,
        input_path=checkpoint_path,
    )

    assert config.checkpoint_path == checkpoint_path


def test_src_does_not_use_dataclasses() -> None:
    for path in Path("src").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imports = {alias.name for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom)) for alias in node.names}
        assert "dataclasses" not in imports, path


def test_src_functions_have_type_annotations() -> None:
    for path in Path("src").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        functions = (node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)))
        for function in functions:
            arguments = [
                *function.args.posonlyargs,
                *function.args.args,
                *function.args.kwonlyargs,
            ]
            missing = [argument.arg for argument in arguments if argument.arg not in {"self", "cls"} and argument.annotation is None]
            if function.args.vararg is not None and function.args.vararg.annotation is None:
                missing.append(f"*{function.args.vararg.arg}")
            if function.args.kwarg is not None and function.args.kwarg.annotation is None:
                missing.append(f"**{function.args.kwarg.arg}")
            assert not missing, f"{path}:{function.lineno}: {missing}"
            assert function.returns is not None, f"{path}:{function.lineno}"
