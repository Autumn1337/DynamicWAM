"""Precompute the exact DOMINO language bank with the pinned Wan UMT5 encoder."""

from __future__ import annotations

import argparse
from pathlib import Path

from dynamicwam.config import load_profile
from dynamicwam.config.schema import OFFICIAL_LEVEL1_TASKS
from dynamicwam.external_assets import verify_wan_assets
from dynamicwam.integrity import DOMINO_RUNTIME_ASSET_PREFIXES, sha256_tree
from dynamicwam.language import (
    assert_language_embeddings_equal,
    generate_domino_language_prompts,
    load_language_embeddings,
)


def _parse_tasks(values: list[str] | None) -> tuple[str, ...]:
    if not values:
        return OFFICIAL_LEVEL1_TASKS
    tasks = tuple(values)
    if len(set(tasks)) != len(tasks):
        raise ValueError("--task values must be unique")
    unknown = sorted(set(tasks) - set(OFFICIAL_LEVEL1_TASKS))
    if unknown:
        raise ValueError(f"unknown DOMINO Level 1 tasks: {unknown}")
    return tasks


def run(*, config_path: str, tasks: tuple[str, ...], device: str) -> None:
    import torch

    from dynamicwam.vendor.wan.modules.t5 import T5EncoderModel

    profile = load_profile(config_path)
    raw = profile.raw
    collection = raw["collection"]
    benchmark = profile.benchmark_config()
    paths = profile.raw["paths"]
    wan_root = Path(paths["wan_root"])
    verify_wan_assets(
        root=wan_root,
        manifest_path=Path(paths["external_assets_manifest"]),
        purpose="language",
    )
    domino_root = Path(benchmark["domino_root"])
    generated_configs = tuple(
        f"task_config/{name}.yml"
        for name in (
            str(collection["clean_config_name"]),
            str(collection["randomized_config_name"]),
        )
    )
    actual_domino_sha256 = sha256_tree(
        domino_root,
        excluded_relative_paths=generated_configs,
        excluded_relative_prefixes=DOMINO_RUNTIME_ASSET_PREFIXES,
    )
    if actual_domino_sha256 != benchmark["domino_source_sha256"]:
        raise RuntimeError(
            "DOMINO source differs from the pinned language-generation source: "
            f"{actual_domino_sha256}"
        )
    output_root = Path(paths["language_embeddings"])
    output_root.mkdir(parents=True, exist_ok=True)

    torch_device = torch.device(device)
    if torch_device.type != "cuda":
        raise ValueError("canonical UMT5 language precomputation requires CUDA")
    torch.cuda.set_device(torch_device)
    encoder = T5EncoderModel(
        text_len=512,
        dtype=torch.bfloat16,
        device=torch_device,
        checkpoint_path=str(wan_root / "models_t5_umt5-xxl-enc-bf16.pth"),
        tokenizer_path=str(wan_root / "google" / "umt5-xxl"),
    )

    for task_index, task in enumerate(OFFICIAL_LEVEL1_TASKS):
        if task not in tasks:
            continue
        scene_info_path = (
            Path(paths["raw_dataset"])
            / "clean"
            / task
            / str(collection["clean_config_name"])
            / "scene_info.json"
        )
        prompts = generate_domino_language_prompts(
            domino_root=domino_root,
            scene_info_path=scene_info_path,
            task=task,
            expected_episodes=int(collection["clean_episodes_per_task"]),
            prompts_per_task=int(raw["data"]["language_prompts_per_task"]),
            seed=int(raw["data"]["language_prompt_seed"]) + task_index,
            scene_prefix=str(raw["inference"]["scene_prefix"]),
        )
        with torch.inference_mode():
            generated = [
                value.detach().cpu().contiguous()
                for value in encoder(prompts, device=torch_device)
            ]
        target = output_root / f"{task}.pt"
        if target.is_file():
            assert_language_embeddings_equal(
                load_language_embeddings(target),
                generated,
                label=str(target),
            )
            print(f"verified existing language bank: {task}", flush=True)
            continue
        temporary = target.with_name(f".{target.name}.tmp")
        try:
            torch.save(generated, temporary)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        load_language_embeddings(target)
        print(f"wrote language bank: {task}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--task", action="append")
    parser.add_argument("--device", default="cuda:0")
    arguments = parser.parse_args()
    run(
        config_path=arguments.config,
        tasks=_parse_tasks(arguments.task),
        device=arguments.device,
    )


if __name__ == "__main__":
    main()
