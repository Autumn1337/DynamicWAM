from __future__ import annotations

from dataclasses import dataclass, field

import torch

from dynamicwam.inference.model_loader import DynamicWAMRuntime
from dynamicwam.vendor.wan.utils.fm import FlowMatchScheduler


@dataclass
class DynamicWAMRunner:
    """Shared DynamicWAM denoising state used by the head-flow runner."""

    runtime: DynamicWAMRuntime
    current_instruction: str = ""
    cached_text_embeddings: list[torch.Tensor] | None = None
    action_scheduler: FlowMatchScheduler = field(init=False)
    video_scheduler: FlowMatchScheduler = field(init=False)

    def __post_init__(self) -> None:
        self.action_scheduler = FlowMatchScheduler(
            shift=self.runtime.flow_match_shift,
            sigma_min=self.runtime.flow_match_sigma_min,
            extra_one_step=self.runtime.flow_match_extra_one_step,
        )
        self.video_scheduler = FlowMatchScheduler(
            shift=self.runtime.flow_match_shift,
            sigma_min=self.runtime.flow_match_sigma_min,
            extra_one_step=self.runtime.flow_match_extra_one_step,
        )
        self._reset_schedulers()

    @property
    def model(self):
        return self.runtime.model

    @property
    def device(self) -> str:
        return self.runtime.device

    def _reset_schedulers(self) -> None:
        self.action_scheduler.set_timesteps(
            num_inference_steps=self.runtime.num_inference_steps,
            training=False,
        )
        self.video_scheduler.set_timesteps(
            num_inference_steps=self.runtime.num_video_inference_steps,
            training=False,
        )

    def reset(self) -> None:
        self.current_instruction = ""
        self.cached_text_embeddings = None
        self._reset_schedulers()

    def set_instruction(self, instruction: str) -> None:
        normalized = str(instruction or "")
        if normalized != self.current_instruction:
            self.current_instruction = normalized
            self.cached_text_embeddings = None

    def _get_text_embeddings(self) -> list[torch.Tensor]:
        if self.cached_text_embeddings is not None:
            return self.cached_text_embeddings
        instruction = f"{self.runtime.scene_prefix}{self.current_instruction}"
        encoded = self.runtime.t5_encoder([instruction], self.device)
        if isinstance(encoded, torch.Tensor):
            if encoded.ndim != 3:
                raise ValueError(f"unexpected T5 output shape: {tuple(encoded.shape)}")
            values = [encoded[0]]
        elif isinstance(encoded, list):
            values = encoded
        else:
            raise TypeError(f"unexpected T5 output type: {type(encoded).__name__}")
        self.cached_text_embeddings = [
            value.to(self.device, dtype=torch.bfloat16) for value in values
        ]
        return self.cached_text_embeddings

    @staticmethod
    def _latent_spatial_size(pixel_size: tuple[int, int]) -> tuple[int, int]:
        downsample = 16
        height, width = (int(value) for value in pixel_size)
        if height % downsample or width % downsample:
            raise ValueError(
                f"future_video_size must be divisible by {downsample}: {pixel_size}"
            )
        return height // downsample, width // downsample

    def _video_refresh_schedule(self) -> dict[int, int]:
        return {
            action_step: video_step
            for video_step, action_step in enumerate(self.runtime.video_refresh_steps)
        }
