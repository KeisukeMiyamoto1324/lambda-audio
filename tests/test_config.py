import ast
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.inference.config import InferenceConfig
from src.stage1.config import Stage1Config
from src.stage2.config import Stage2Config


def test_training_config_validates_positive_values() -> None:
    with pytest.raises(ValidationError):
        Stage1Config(num_workers=0)


def test_training_config_is_frozen() -> None:
    config = Stage1Config()

    with pytest.raises(ValidationError):
        config.max_steps = 1  # type: ignore[misc]


def test_stage2_config_validates_checkpoint_path(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        Stage2Config(initialization_checkpoint=tmp_path / "missing.ckpt")


def test_stage2_resume_does_not_require_initialization_checkpoint(
    tmp_path: Path,
) -> None:
    resume_checkpoint = tmp_path / "last.ckpt"
    resume_checkpoint.touch()

    config = Stage2Config(
        initialization_checkpoint=tmp_path / "missing.ckpt",
        resume_checkpoint=resume_checkpoint,
    )

    assert config.resume_checkpoint == resume_checkpoint


def test_inference_config_validates_paths(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "model.ckpt"
    checkpoint_path.touch()

    config = InferenceConfig(
        checkpoint_path=checkpoint_path,
        input_dir=tmp_path,
    )

    assert config.checkpoint_path == checkpoint_path


def test_src_does_not_use_dataclasses() -> None:
    for path in Path("src").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imports = {
            alias.name
            for node in tree.body
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        assert "dataclasses" not in imports, path


def test_src_functions_have_type_annotations() -> None:
    for path in Path("src").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        functions = (
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        )
        for function in functions:
            arguments = [
                *function.args.posonlyargs,
                *function.args.args,
                *function.args.kwonlyargs,
            ]
            missing = [
                argument.arg
                for argument in arguments
                if argument.arg not in {"self", "cls"} and argument.annotation is None
            ]
            if (
                function.args.vararg is not None
                and function.args.vararg.annotation is None
            ):
                missing.append(f"*{function.args.vararg.arg}")
            if (
                function.args.kwarg is not None
                and function.args.kwarg.annotation is None
            ):
                missing.append(f"**{function.args.kwarg.arg}")
            assert not missing, f"{path}:{function.lineno}: {missing}"
            assert function.returns is not None, f"{path}:{function.lineno}"
