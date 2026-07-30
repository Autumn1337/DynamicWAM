"""Stage 3 DynamicWAM joint video-action refinement entry."""

from __future__ import annotations

from dynamicwam.training.train.video_action_trainer import run_video_action_training


def main() -> None:
    run_video_action_training(
        description="Stage 3 DynamicWAM joint video-action refinement",
        stage_name="stage3",
    )


if __name__ == "__main__":
    main()
