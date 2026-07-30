from __future__ import annotations

import numpy as np
import torch

from dynamicwam.inference.runner import DynamicWAMRunner


def flow_frames_to_pixels(
    flow_frames: np.ndarray,
    *,
    count: int,
    height: int,
    width: int,
    device,
    dtype,
) -> torch.Tensor:
    """Convert four uint8 RGB flow frames to ``[1,3,4,H,W]`` in [-1, 1]."""
    import cv2

    if (
        flow_frames.dtype != np.uint8
        or flow_frames.ndim != 4
        or flow_frames.shape[0] != count
        or flow_frames.shape[-1] != 3
    ):
        raise ValueError(
            f"DynamicWAM requires flow_frames with shape [{count},H,W,3] and dtype uint8, "
            f"got {flow_frames.shape} {flow_frames.dtype}"
        )
    resized = np.stack(
        [
            cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
            for frame in flow_frames
        ],
        axis=0,
    )
    tensor = (
        torch.from_numpy(resized)
        .permute(3, 0, 1, 2)
        .unsqueeze(0)
        .to(device=device, dtype=dtype)
    )
    return tensor.div_(127.5).sub_(1.0)


class HeadFlowRunner(DynamicWAMRunner):
    """DynamicWAM full-chunk sampler: head-flow condition, WAM, 10 steps, refresh 0/1."""

    def _initialize_video_latent(
        self,
        first_frame: torch.Tensor,
        flow_frames: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        video_dtype = self.model.compact_wan.video_model.precision
        first_frame = first_frame.to(self.device, dtype=video_dtype)
        frame_pixels = first_frame.mul(2.0).sub(1.0).unsqueeze(2)
        flow_pixels = flow_frames_to_pixels(
            flow_frames,
            count=int(self.runtime.head_flow_config["count"]),
            height=first_frame.shape[-2],
            width=first_frame.shape[-1],
            device=self.device,
            dtype=video_dtype,
        )
        condition_pixels = torch.cat([flow_pixels, frame_pixels], dim=2)
        with torch.inference_mode():
            condition_latent = self.model.compact_wan.encode_video(condition_pixels)
        future_size = self.model.compact_wan.config.future_video_size
        latent_height, latent_width = self._latent_spatial_size(future_size)
        future_latent = torch.randn(
            (
                1,
                int(condition_latent.shape[1]),
                self.runtime.num_video_frames // 4,
                latent_height,
                latent_width,
            ),
            device=self.device,
            dtype=video_dtype,
        )
        return future_latent, condition_latent

    @torch.inference_mode()
    def sample_chunk(
        self,
        *,
        first_frame: torch.Tensor,
        state: torch.Tensor,
        flow_frames: np.ndarray,
        motion_features: np.ndarray,
        motion_interval_valid_mask: np.ndarray,
        motion_acceleration_valid_mask: np.ndarray,
    ) -> torch.Tensor:
        video_dtype = self.model.compact_wan.video_model.precision
        action_dtype = next(self.model.action_expert.parameters()).dtype
        normalized_state = self.runtime.action_normalizer.normalize(
            state.to(self.device, dtype=action_dtype)
        )
        text_embeddings = self._get_text_embeddings()
        video_latent, condition_latent = self._initialize_video_latent(
            first_frame,
            flow_frames,
        )
        motion_batch = {
            "absolute_motion_features": torch.as_tensor(
                motion_features,
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0),
            "absolute_motion_interval_valid_mask": torch.as_tensor(
                motion_interval_valid_mask,
                dtype=torch.bool,
                device=self.device,
            ).unsqueeze(0),
            "absolute_motion_acceleration_valid_mask": torch.as_tensor(
                motion_acceleration_valid_mask,
                dtype=torch.bool,
                device=self.device,
            ).unsqueeze(0),
        }
        noisy_actions = torch.randn(
            (1, self.runtime.chunk_size, self.model.config.action_dim),
            device=self.device,
            dtype=action_dtype,
        )
        action_timesteps = self.action_scheduler.timesteps.to(
            device=self.device,
            dtype=action_dtype,
        )
        video_timesteps = self.video_scheduler.timesteps.to(
            device=self.device,
            dtype=video_dtype,
        )
        refresh_schedule = self._video_refresh_schedule()
        video_cache = None

        for action_step in range(self.runtime.num_inference_steps):
            action_t = action_timesteps[action_step].expand(1)
            video_step = refresh_schedule.get(action_step)
            if video_step is not None:
                video_t = video_timesteps[video_step].expand(1)
                outputs = self.model.forward_with_video_cache(
                    {
                        "video_t": video_t,
                        "initial_state": normalized_state,
                        "noisy_actions": noisy_actions,
                        "action_t": action_t,
                        "text_embeddings": text_embeddings,
                        "condition_latent": condition_latent,
                        "future_latent": video_latent,
                        **motion_batch,
                    }
                )
                video_cache = outputs["video_cache"]
                video_latent = self.video_scheduler.step(
                    outputs["video_pred"],
                    video_t,
                    video_latent,
                )
            else:
                if video_cache is None:
                    raise RuntimeError(
                        "video_refresh_steps must build a cache at step 0"
                    )
                outputs = self.model.forward_action_with_video_cache(
                    {
                        "video_cache": video_cache,
                        "initial_state": normalized_state,
                        "noisy_actions": noisy_actions,
                        "action_t": action_t,
                        **motion_batch,
                    }
                )
            noisy_actions = self.action_scheduler.step(
                outputs["action_pred"],
                action_t,
                noisy_actions,
            )
        return self.runtime.action_normalizer.denormalize(noisy_actions)
