"""DOMINO instruction generation and UMT5 embedding contracts."""

from __future__ import annotations

import importlib.util
import json
import random
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from dynamicwam.config.schema import OFFICIAL_LEVEL1_TASKS

if TYPE_CHECKING:
    import torch

LANGUAGE_PROMPTS_PER_TASK = 100
DOMINO_INSTRUCTION_GENERATOR = Path(
    "description/utils/generate_episode_instructions.py"
)
UNRESOLVED_PLACEHOLDER = re.compile(r"\{[^{}]+\}")


def _load_domino_instruction_generator(domino_root: Path) -> Any:
    path = domino_root / DOMINO_INSTRUCTION_GENERATOR
    if not path.is_file():
        raise FileNotFoundError(f"DOMINO instruction generator is missing: {path}")
    spec = importlib.util.spec_from_file_location(
        "_dynamicwam_domino_instruction_generator",
        path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load DOMINO instruction generator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    generator = getattr(module, "generate_episode_descriptions", None)
    if not callable(generator):
        raise AttributeError(
            "DOMINO instruction module has no callable generate_episode_descriptions"
        )
    return generator


def generate_domino_language_prompts(
    *,
    domino_root: Path,
    scene_info_path: Path,
    task: str,
    expected_episodes: int,
    prompts_per_task: int,
    seed: int,
    scene_prefix: str,
) -> list[str]:
    """Generate one deterministic task-level bank from pinned DOMINO inputs."""

    if task not in OFFICIAL_LEVEL1_TASKS:
        raise ValueError(f"unknown DOMINO Level 1 task: {task}")
    if (
        isinstance(expected_episodes, bool)
        or not isinstance(expected_episodes, int)
        or expected_episodes <= 0
    ):
        raise ValueError("expected_episodes must be a positive integer")
    if prompts_per_task != LANGUAGE_PROMPTS_PER_TASK:
        raise ValueError(
            "DynamicWAM requires exactly "
            f"{LANGUAGE_PROMPTS_PER_TASK} language prompts per task"
        )
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("language prompt seed must be a non-negative integer")
    if not isinstance(scene_prefix, str) or not scene_prefix.endswith(": "):
        raise ValueError("scene_prefix must be a string ending in ': '")
    if not scene_info_path.is_file():
        raise FileNotFoundError(
            f"DOMINO collection scene info is missing: {scene_info_path}"
        )

    scene_info = json.loads(scene_info_path.read_text(encoding="utf-8"))
    if not isinstance(scene_info, dict):
        raise TypeError(f"DOMINO scene info must be a mapping: {scene_info_path}")
    expected_keys = {
        f"episode_{episode_index}" for episode_index in range(expected_episodes)
    }
    if set(scene_info) != expected_keys:
        raise ValueError(
            f"DOMINO scene set differs for {task}: "
            f"missing={sorted(expected_keys - set(scene_info))}, "
            f"unknown={sorted(set(scene_info) - expected_keys)}"
        )
    representative = scene_info["episode_0"]
    if not isinstance(representative, dict) or not isinstance(
        representative.get("info"),
        dict,
    ):
        raise ValueError(
            f"DOMINO episode_0 has no instruction parameters: {scene_info_path}"
        )

    generator = _load_domino_instruction_generator(domino_root)
    random_state = random.getstate()
    random.seed(seed)
    try:
        generated = generator(
            task,
            [representative["info"]],
            prompts_per_task,
        )
    finally:
        random.setstate(random_state)
    if (
        not isinstance(generated, list)
        or len(generated) != 1
        or not isinstance(generated[0], dict)
    ):
        raise RuntimeError(f"DOMINO returned an invalid prompt payload for {task}")
    seen = generated[0].get("seen")
    if (
        not isinstance(seen, list)
        or len(seen) != prompts_per_task
        or any(
            not isinstance(prompt, str)
            or not prompt
            or "\n" in prompt
            or UNRESOLVED_PLACEHOLDER.search(prompt)
            for prompt in seen
        )
    ):
        raise RuntimeError(
            f"DOMINO did not generate {prompts_per_task} resolved prompts for {task}"
        )
    return [f"{scene_prefix}{prompt}" for prompt in seen]


def load_language_embeddings(path: Path) -> list[torch.Tensor]:
    """Load and validate one exact per-task language embedding bank."""

    import torch

    if not path.is_file():
        raise FileNotFoundError(f"language embedding bank is missing: {path}")
    payload: Any = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, list):
        raise ValueError(f"language embedding bank must contain a list: {path}")
    if len(payload) != LANGUAGE_PROMPTS_PER_TASK:
        raise ValueError(
            f"{path} must contain exactly {LANGUAGE_PROMPTS_PER_TASK} embeddings"
        )
    embeddings: list[torch.Tensor] = []
    for index, value in enumerate(payload):
        if not isinstance(value, torch.Tensor) or value.ndim != 2:
            shape = tuple(value.shape) if isinstance(value, torch.Tensor) else None
            raise ValueError(
                f"{path} embedding {index} must be a [tokens, dim] tensor, "
                f"got {type(value).__name__} with shape {shape}"
            )
        if value.shape[0] <= 0 or value.shape[1] <= 0:
            raise ValueError(f"{path} embedding {index} must be non-empty")
        if not value.is_floating_point():
            raise ValueError(f"{path} embedding {index} must be floating point")
        embeddings.append(value.detach().cpu().contiguous())
    return embeddings


def assert_language_embeddings_equal(
    left: list[torch.Tensor],
    right: list[torch.Tensor],
    *,
    label: str,
) -> None:
    """Reject an existing bank unless every tensor is exactly equal."""

    import torch

    if len(left) != len(right) or any(
        not torch.equal(left_value, right_value)
        for left_value, right_value in zip(left, right, strict=True)
    ):
        raise RuntimeError(f"language embedding bank differs: {label}")
