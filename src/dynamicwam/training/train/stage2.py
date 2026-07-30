"""Stage 2 DynamicWAM frozen-video action training entry."""

from __future__ import annotations

from dynamicwam.training.train.video_action_trainer import run_video_action_training


def main() -> None:
    run_video_action_training(
        description="Stage 2 DynamicWAM frozen-video action training",
        stage_name="stage2",
    )


if __name__ == "__main__":
    main()
