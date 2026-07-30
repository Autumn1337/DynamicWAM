from __future__ import annotations

from typing import List

import torch


def build_packed_pca_sample_ids(
    dataset,
    *,
    requested_episodes: int,
    samples_per_episode: int,
    seed: int,
) -> List[int]:
    """Select a balanced, deterministic PCA subset from a packed train dataset."""
    if samples_per_episode <= 0:
        raise ValueError(
            f"PCA samples_per_episode must be positive, got {samples_per_episode}"
        )
    episodes = list(getattr(dataset, "episodes_with_samples", []))
    if not episodes:
        raise ValueError("packed train dataset has no episodes for PCA")
    requested_count = int(requested_episodes)
    if requested_count <= 0 or requested_count > len(episodes):
        raise ValueError(
            "invalid PCA episode count: "
            f"requested={requested_count}, available={len(episodes)}"
        )

    sample_ids: List[int] = []
    episode_generator = torch.Generator(device="cpu")
    episode_generator.manual_seed(int(seed) + 10_007)
    selected_slots = torch.randperm(
        len(episodes),
        generator=episode_generator,
    )[:requested_count]
    for episode_slot in selected_slots.tolist():
        episode = episodes[int(episode_slot)]
        first_sample_id = int(episode["first_sample_id"])
        sample_count = int(episode["sample_count"])
        episode_id = int(episode["episode_id"])
        if sample_count < samples_per_episode:
            raise ValueError(
                f"PCA episode {episode_id} has only {sample_count} unique "
                f"samples; requires {samples_per_episode}"
            )
        sample_generator = torch.Generator(device="cpu")
        sample_generator.manual_seed(int(seed) * 1_000_003 + episode_id)
        local_indices = torch.randperm(
            sample_count,
            generator=sample_generator,
        )[:samples_per_episode]
        sample_ids.extend(
            first_sample_id + int(index) for index in local_indices.tolist()
        )
    shuffle_generator = torch.Generator(device="cpu")
    shuffle_generator.manual_seed(int(seed) + 97_531)
    order = torch.randperm(len(sample_ids), generator=shuffle_generator)
    return [sample_ids[int(index)] for index in order.tolist()]
