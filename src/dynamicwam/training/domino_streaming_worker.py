"""Planner and renderer workers for streaming exact-time DOMINO collection."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import pickle
import sys
import time
import traceback
from copy import deepcopy
from pathlib import Path
from typing import Any

from .domino_stream_queue import (
    RenderClaim,
    SeedClaim,
    StreamQueue,
    atomic_copy,
    atomic_write_bytes,
    sha256_file,
    validate_hdf5_episode,
)


def _load_domino_arguments(
    *,
    domino_root: Path,
    task: str,
    task_config: str,
    job_root: Path,
) -> dict[str, Any]:
    import yaml
    from envs._GLOBAL_CONFIGS import CONFIGS_PATH

    config_path = domino_root / "task_config" / f"{task_config}.yml"
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"DOMINO task config is not a mapping: {config_path}")
    payload["task_name"] = task
    payload["task_config"] = task_config
    payload["save_path"] = str(job_root)

    embodiment = payload.get("embodiment")
    if embodiment != ["aloha-agilex"]:
        raise RuntimeError("Streaming absolute-motion collection requires aloha-agilex")
    embodiment_types = yaml.safe_load(
        (Path(CONFIGS_PATH) / "_embodiment_config.yml").read_text(encoding="utf-8")
    )
    robot_file = embodiment_types[embodiment[0]]["file_path"]
    robot_config = yaml.safe_load(
        (Path(robot_file) / "config.yml").read_text(encoding="utf-8")
    )
    payload.update(
        {
            "left_robot_file": robot_file,
            "right_robot_file": robot_file,
            "dual_arm_embodied": True,
            "left_embodiment_config": robot_config,
            "right_embodiment_config": robot_config,
            "embodiment_name": embodiment[0],
            "render_freq": 0,
        }
    )
    return payload


def _task_environment(task: str) -> Any:
    module = importlib.import_module(f"envs.{task}")
    task_class = getattr(module, task)
    return task_class()


def _install_replay_without_planners() -> None:
    import numpy as np
    from envs._base_task import Base_Task
    from envs.robot.robot import Robot

    class ReplayGripperPlanner:
        @staticmethod
        def plan_grippers(now_val: float, target_val: float) -> dict[str, Any]:
            num_step = 200
            return {
                "num_step": num_step,
                "per_step": (target_val - now_val) / num_step,
                "result": np.linspace(now_val, target_val, num_step),
            }

        def __getattr__(self, name: str) -> Any:
            raise RuntimeError(
                "Replay attempted to invoke motion planning method "
                f"{name!r}; trajectories must supply every arm path"
            )

    def no_planner(self: Any, scene: Any = None) -> None:
        del scene
        self.communication_flag = False
        self.left_planner = ReplayGripperPlanner()
        self.right_planner = ReplayGripperPlanner()
        self.left_mplib_planner = None
        self.right_mplib_planner = None

    def choose_replay_grasp_pose(
        self: Any,
        res_pose: Any,
        center_pose: Any,
        arm_tag: Any = None,
    ) -> Any:
        if self.need_plan:
            raise RuntimeError(
                "Renderer attempted to select a new grasp with need_plan=True"
            )
        target_poses = self.robot.create_target_pose_list(
            res_pose,
            center_pose,
            arm_tag,
        )
        if not target_poses:
            self.plan_success = False
            return None
        return deepcopy(target_poses[0])

    Robot.set_planner = no_planner
    Base_Task.choose_best_pose = choose_replay_grasp_pose


def _install_lightweight_planner_environment() -> None:
    import numpy as np
    import sapien.core as sapien
    from envs._base_task import Base_Task

    class NoCaptureCameras:
        @staticmethod
        def update_wrist_camera(left_pose: Any, right_pose: Any) -> None:
            del left_pose, right_pose

    class PlannerLight:
        def __init__(self, color: Any):
            self.color = np.asarray(color, dtype=np.float64)

        def set_color(self, color: Any) -> None:
            self.color = np.asarray(color, dtype=np.float64)

    def setup_scene_without_renderer(
        self: Any,
        **arguments: Any,
    ) -> None:
        if self.render_freq:
            raise RuntimeError(
                "Planner-only DOMINO environments cannot create a viewer"
            )
        self.engine = sapien.Engine()
        scene_configuration = sapien.SceneConfig()
        self.scene = self.engine.create_scene(scene_configuration)
        self.scene.set_timestep(arguments.get("timestep", 1 / 250))
        self.scene.add_ground(arguments.get("ground_height", 0))
        self.scene.default_physical_material = self.scene.create_physical_material(
            arguments.get("static_friction", 0.5),
            arguments.get("dynamic_friction", 0.5),
            arguments.get("restitution", 0),
        )
        self._planner_ambient_light = np.asarray(
            arguments.get("ambient_light", [0.5, 0.5, 0.5]),
            dtype=np.float64,
        )
        self.direction_light_lst = []
        direction_lights = arguments.get(
            "direction_lights",
            [[[0, 0.5, -1], [0.5, 0.5, 0.5]]],
        )
        for direction_light in direction_lights:
            if self.random_light:
                direction_light[1] = [
                    np.random.rand(),
                    np.random.rand(),
                    np.random.rand(),
                ]
            self.direction_light_lst.append(PlannerLight(direction_light[1]))
        self.point_light_lst = []
        point_lights = arguments.get(
            "point_lights",
            [
                [[1, 0, 1.8], [1, 1, 1]],
                [[-1, 0, 1.8], [1, 1, 1]],
            ],
        )
        for point_light in point_lights:
            if self.random_light:
                point_light[1] = [
                    np.random.rand(),
                    np.random.rand(),
                    np.random.rand(),
                ]
            self.point_light_lst.append(PlannerLight(point_light[1]))

    def load_camera_without_capture(self: Any, **arguments: Any) -> None:
        static_camera_information = arguments["left_embodiment_config"][
            "static_camera_list"
        ]
        collect_head_camera = arguments["camera"].get(
            "collect_head_camera",
            True,
        )
        for camera_info in static_camera_information:
            if camera_info.get("forward") is None:
                camera_info["forward"] = (
                    -1 * np.asarray(camera_info["position"])
                ).tolist()
            if camera_info.get("left") is None:
                camera_info["left"] = [
                    -camera_info["forward"][1],
                    camera_info["forward"][0],
                    0,
                ]
            if camera_info["name"] == "head_camera" and not collect_head_camera:
                continue
            vector = np.random.randn(3)
            _random_direction = vector / np.linalg.norm(vector)
            head_distance = (
                self.random_head_camera_dis
                if camera_info["name"] == "head_camera"
                else 0
            )
            np.random.uniform(low=0, high=head_distance)

        self.cameras = NoCaptureCameras()
        self._update_kinematic_tasks()
        self.scene.step()

    def update_without_rendering(self: Any) -> None:
        if self.crazy_random_light:
            for light in self.point_light_lst:
                light.set_color([np.random.rand(), np.random.rand(), np.random.rand()])
            for light in self.direction_light_lst:
                light.set_color([np.random.rand(), np.random.rand(), np.random.rand()])
            self._planner_ambient_light = np.clip(
                self._planner_ambient_light + np.random.rand(3) * 0.2 - 0.1,
                0,
                1,
            )

    Base_Task.setup_scene = setup_scene_without_renderer
    Base_Task.load_camera = load_camera_without_capture
    Base_Task._update_render = update_without_rendering


def _install_renderer_capture_optimizations() -> None:
    import numpy as np
    import torch
    from envs._base_task import Base_Task
    from envs.camera.camera import Camera

    original_update_picture = Camera.update_picture

    def update_picture_profiled(self: Any) -> None:
        started = time.perf_counter()
        original_update_picture(self)
        self._absolute_motion_take_picture_seconds = (
            getattr(self, "_absolute_motion_take_picture_seconds", 0.0)
            + time.perf_counter()
            - started
        )

    def get_rgb_batched(self: Any) -> dict[str, dict[str, Any]]:
        started = time.perf_counter()
        cameras: list[tuple[str, Any]] = []
        if self.collect_wrist_camera:
            cameras.extend(
                (
                    ("left_camera", self.left_camera),
                    ("right_camera", self.right_camera),
                )
            )
        for camera, camera_name in zip(
            self.static_camera_list,
            self.static_camera_name,
            strict=True,
        ):
            if camera_name != "head_camera" or self.collect_head_camera:
                cameras.append((camera_name, camera))
        if not cameras:
            raise RuntimeError("Exact DOMINO capture has no RGB cameras")

        tensors = [
            camera.get_picture_cuda("Color").torch()[..., :3]
            for _name, camera in cameras
        ]
        pixel_counts = [int(tensor.shape[0] * tensor.shape[1]) for tensor in tensors]
        packed = torch.cat(
            [tensor.reshape(-1, 3) for tensor in tensors],
            dim=0,
        )
        host = packed.mul(255).clamp(0, 255).to(torch.uint8).cpu().numpy()

        result: dict[str, dict[str, Any]] = {}
        offset = 0
        for (name, _camera), tensor, count in zip(
            cameras,
            tensors,
            pixel_counts,
            strict=True,
        ):
            result[name] = {
                "rgb": host[offset : offset + count].reshape(
                    int(tensor.shape[0]),
                    int(tensor.shape[1]),
                    3,
                )
            }
            offset += count
        self._absolute_motion_readback_seconds = (
            getattr(self, "_absolute_motion_readback_seconds", 0.0)
            + time.perf_counter()
            - started
        )
        return result

    def update_render_once_for_capture(self: Any) -> None:
        if self.crazy_random_light:
            for light in self.point_light_lst:
                light.set_color([np.random.rand(), np.random.rand(), np.random.rand()])
            for light in self.direction_light_lst:
                light.set_color([np.random.rand(), np.random.rand(), np.random.rand()])
            ambient = np.clip(
                np.asarray(self.scene.ambient_light) + np.random.rand(3) * 0.2 - 0.1,
                0,
                1,
            )
            self.scene.set_ambient_light(ambient)
        self.cameras.update_wrist_camera(
            self.robot.left_camera.get_pose(),
            self.robot.right_camera.get_pose(),
        )
        if getattr(self, "_absolute_motion_capture_sync", False):
            started = time.perf_counter()
            self.scene.update_render()
            self._absolute_motion_scene_update_seconds = (
                getattr(
                    self,
                    "_absolute_motion_scene_update_seconds",
                    0.0,
                )
                + time.perf_counter()
                - started
            )

    def take_picture_in_memory(self: Any) -> None:
        if not self.save_data:
            return
        if self.FRAME_IDX == 0:
            self._absolute_motion_frames = []
            self._absolute_motion_scene_update_seconds = 0.0
            self.cameras._absolute_motion_take_picture_seconds = 0.0
            self.cameras._absolute_motion_readback_seconds = 0.0

        self._absolute_motion_capture_sync = True
        try:
            frame = self.get_obs()
        finally:
            self._absolute_motion_capture_sync = False
        if self.data_type.get("interception", False):
            frame["interception"] = self._build_interception_frame_data(
                frame["observation"]
            )
        self._absolute_motion_frames.append(frame)
        self.FRAME_IDX += 1

    Camera.update_picture = update_picture_profiled
    Camera.get_rgb = get_rgb_batched
    Base_Task._update_render = update_render_once_for_capture
    Base_Task._take_picture = take_picture_in_memory


def _post_play_success_contract(task: str, environment: Any) -> None:
    if (
        task == "dump_bin_bigbin"
        and getattr(environment, "use_dynamic", False)
        and getattr(environment, "plan_success", False)
    ):
        environment.verify_dynamic_lift()


def _is_retryable_render_resource_failure(message: str) -> bool:
    normalized = message.lower()
    return any(
        marker in normalized
        for marker in (
            "cannot create buffer",
            "out of memory",
            "cuda error: memory allocation",
            "cudaerrormemoryallocation",
            "vk_error_out_of_device_memory",
            "vk_error_out_of_host_memory",
            "erroroutofdevicememory",
            "erroroutofhostmemory",
        )
    )


def _trajectory_payload(environment: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "left_joint_path": deepcopy(environment.left_joint_path),
        "right_joint_path": deepcopy(environment.right_joint_path),
    }
    dynamic_info = getattr(environment, "_saved_dynamic_motion_info", None)
    if dynamic_info is not None:
        copied = deepcopy(dynamic_info)
        trajectory_params = copied.get("trajectory_params")
        if (
            isinstance(trajectory_params, dict)
            and trajectory_params.get("type") == "trajectory"
        ):
            trajectory_params.pop("trajectory_func", None)
        payload["dynamic_motion_info"] = copied
    return payload


def _write_trajectory_payload(
    queue: StreamQueue,
    *,
    seed: int,
    payload: dict[str, Any],
) -> tuple[Path, str]:
    path = queue.item_root / f"planned_seed_{seed}.pkl"
    atomic_write_bytes(
        path,
        pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL),
    )
    return path, sha256_file(path)


def _load_trajectory(
    claim: RenderClaim,
) -> dict[str, Any]:
    if (
        not claim.trajectory_path.is_file()
        or sha256_file(claim.trajectory_path) != claim.trajectory_sha256
    ):
        raise RuntimeError(f"Trajectory payload changed for seed {claim.seed}")
    with claim.trajectory_path.open("rb") as stream:
        payload = pickle.load(stream)
    if not isinstance(payload, dict):
        raise TypeError(f"Trajectory payload is not a mapping: {claim.trajectory_path}")
    required = {"left_joint_path", "right_joint_path"}
    if not required.issubset(payload):
        raise RuntimeError(
            f"Trajectory payload is missing paths: {claim.trajectory_path}"
        )
    return payload


def _package_frames_to_hdf5(
    *,
    frames: list[dict[str, Any]],
    temporary_hdf5: Path,
) -> tuple[int, str]:
    import h5py
    import numpy as np
    from envs.utils import pkl2hdf5

    if not frames:
        raise RuntimeError("In-memory frame cache is empty")
    data = pkl2hdf5.parse_dict_structure(frames[0])
    for frame in frames:
        pkl2hdf5.append_data_to_structure(
            data,
            frame,
        )
    pkl2hdf5.validate_leaf_lengths(data, len(frames))
    pkl2hdf5.validate_interception_frames(data, len(frames))

    cache_step = np.asarray(data["interception"]["sim_step_index"])
    cache_time = np.asarray(data["interception"]["sim_time_seconds"])
    temporary_hdf5.parent.mkdir(parents=True, exist_ok=True)
    temporary_hdf5.unlink(missing_ok=True)
    try:
        with h5py.File(temporary_hdf5, "w") as handle:
            pkl2hdf5.create_hdf5_from_dict(handle, data)
        descriptor = os.open(temporary_hdf5, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        episode_info = validate_hdf5_episode(temporary_hdf5)
        with h5py.File(temporary_hdf5, "r") as handle:
            hdf5_step = np.asarray(handle["interception/sim_step_index"])
            hdf5_time = np.asarray(handle["interception/sim_time_seconds"])
        if not np.array_equal(cache_step, hdf5_step) or not np.array_equal(
            cache_time, hdf5_time
        ):
            raise RuntimeError("Exact simulator clock changed while packaging HDF5")
        return episode_info.frame_count, episode_info.sha256
    except Exception:
        temporary_hdf5.unlink(missing_ok=True)
        raise


def _close_environment(
    environment: Any,
    *,
    clear_cache: bool,
) -> None:
    try:
        environment.close_env(clear_cache=clear_cache)
    except Exception:
        traceback.print_exc()


def _queue_from_arguments(arguments: argparse.Namespace) -> StreamQueue:
    return StreamQueue(
        job_root=Path(arguments.job_root),
        task=arguments.task,
        split=arguments.split,
        task_config=arguments.task_config,
        target_episodes=arguments.target_episodes,
    )


def run_planner(arguments: argparse.Namespace) -> None:
    domino_root = Path(arguments.domino_root).resolve()
    job_root = Path(arguments.job_root).resolve()
    queue = _queue_from_arguments(arguments)
    queue.open_existing()
    _install_lightweight_planner_environment()
    base_arguments = _load_domino_arguments(
        domino_root=domino_root,
        task=arguments.task,
        task_config=arguments.task_config,
        job_root=job_root,
    )
    environment = _task_environment(arguments.task)
    worker_id = arguments.worker_id
    try:
        while True:
            status = queue.status()
            if status.complete:
                break
            claim = queue.claim_seed(
                worker_id=worker_id,
                ready_buffer_episodes=arguments.ready_buffer_episodes,
                max_new_attempts=arguments.max_new_attempts,
            )
            if claim is None:
                time.sleep(arguments.poll_seconds)
                continue
            _run_seed_attempt(
                queue=queue,
                environment=environment,
                base_arguments=base_arguments,
                task=arguments.task,
                claim=claim,
            )
    finally:
        if hasattr(environment, "release_episode_resources"):
            environment.release_episode_resources()


def _run_seed_attempt(
    *,
    queue: StreamQueue,
    environment: Any,
    base_arguments: dict[str, Any],
    task: str,
    claim: SeedClaim,
) -> None:
    setup_seconds = 0.0
    play_seconds = 0.0
    success_check_seconds = 0.0
    plan_success = False
    check_success = False
    failure_kind = "success_check"
    failure_message = ""
    try:
        task_arguments = deepcopy(base_arguments)
        task_arguments.update(
            {
                "need_plan": True,
                "save_data": False,
                "left_joint_path": [],
                "right_joint_path": [],
            }
        )
        started = time.perf_counter()
        try:
            environment.setup_demo(
                now_ep_num=0,
                seed=claim.seed,
                **task_arguments,
            )
        finally:
            setup_seconds = time.perf_counter() - started

        started = time.perf_counter()
        try:
            environment.play_once()
            _post_play_success_contract(task, environment)
        finally:
            play_seconds = time.perf_counter() - started
        plan_success = bool(environment.plan_success)

        started = time.perf_counter()
        try:
            check_success = bool(environment.check_success() if plan_success else False)
        finally:
            success_check_seconds = time.perf_counter() - started
        if plan_success and check_success:
            trajectory_path, trajectory_sha256 = _write_trajectory_payload(
                queue,
                seed=claim.seed,
                payload=_trajectory_payload(environment),
            )
            queue.complete_seed_success(
                claim=claim,
                trajectory_path=trajectory_path,
                trajectory_sha256=trajectory_sha256,
                setup_seconds=setup_seconds,
                play_seconds=play_seconds,
                success_check_seconds=success_check_seconds,
            )
            print(
                json.dumps(
                    {
                        "event": "seed_success",
                        "seed": claim.seed,
                        "worker_id": claim.worker_id,
                        "setup_seconds": setup_seconds,
                        "play_seconds": play_seconds,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            return
        if not plan_success:
            failure_kind = "plan_failure"
            failure_message = "environment.plan_success is false"
        else:
            failure_kind = "success_check_failure"
            failure_message = "environment.check_success returned false"
    except Exception as error:
        failure_kind = f"exception:{type(error).__name__}"
        failure_message = str(error)
        traceback.print_exc()
    finally:
        _close_environment(environment, clear_cache=False)

    queue.complete_seed_failure(
        claim=claim,
        setup_seconds=setup_seconds,
        play_seconds=play_seconds,
        success_check_seconds=success_check_seconds,
        plan_success=plan_success,
        check_success=check_success,
        failure_kind=failure_kind,
        failure_message=failure_message,
    )
    print(
        json.dumps(
            {
                "event": "seed_failure",
                "seed": claim.seed,
                "worker_id": claim.worker_id,
                "failure_kind": failure_kind,
                "failure_message": failure_message,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def run_renderer(arguments: argparse.Namespace) -> None:
    domino_root = Path(arguments.domino_root).resolve()
    job_root = Path(arguments.job_root).resolve()
    queue = _queue_from_arguments(arguments)
    queue.open_existing()
    recovered_claims = queue.recover_renderer_claims(worker_id=arguments.worker_id)
    if recovered_claims:
        print(
            json.dumps(
                {
                    "event": "renderer_claim_recovered",
                    "worker_id": arguments.worker_id,
                    "claims": recovered_claims,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    _install_replay_without_planners()
    _install_renderer_capture_optimizations()
    base_arguments = _load_domino_arguments(
        domino_root=domino_root,
        task=arguments.task,
        task_config=arguments.task_config,
        job_root=job_root,
    )
    environment = _task_environment(arguments.task)
    worker_id = arguments.worker_id
    attempts_since_recycle = 0
    try:
        while True:
            status = queue.status()
            if status.complete:
                break
            claim = queue.claim_render(worker_id=worker_id)
            if claim is None:
                time.sleep(arguments.poll_seconds)
                continue
            _run_render_attempt(
                queue=queue,
                environment=environment,
                base_arguments=base_arguments,
                task=arguments.task,
                claim=claim,
            )
            attempts_since_recycle += 1
            if attempts_since_recycle >= arguments.renderer_recycle_attempts:
                _exec_renderer(
                    reason="periodic",
                    attempts=attempts_since_recycle,
                    worker_id=worker_id,
                )
    finally:
        if hasattr(environment, "release_episode_resources"):
            environment.release_episode_resources()


def _run_render_attempt(
    *,
    queue: StreamQueue,
    environment: Any,
    base_arguments: dict[str, Any],
    task: str,
    claim: RenderClaim,
) -> None:
    job_root = queue.job_root
    final_hdf5 = job_root / "data" / f"episode{claim.output_index}.hdf5"
    temporary_hdf5 = final_hdf5.with_name(
        f".{final_hdf5.name}.stream.tmp.{os.getpid()}"
    )
    canonical_trajectory = job_root / "_traj_data" / f"episode{claim.output_index}.pkl"
    temporary_hdf5.unlink(missing_ok=True)
    replay_setup_seconds = 0.0
    replay_seconds = 0.0
    success_check_seconds = 0.0
    package_seconds = 0.0
    publishing_started = False
    failure_kind = "render_failure"
    failure_message = ""
    environment_closed = False
    environment._absolute_motion_frames = []
    try:
        trajectory = _load_trajectory(claim)
        replay_arguments = deepcopy(base_arguments)
        replay_arguments.update(
            {
                "need_plan": False,
                "save_data": True,
                "left_joint_path": [],
                "right_joint_path": [],
            }
        )
        started = time.perf_counter()
        try:
            environment.setup_demo(
                now_ep_num=claim.output_index,
                seed=claim.seed,
                **replay_arguments,
            )
        finally:
            replay_setup_seconds = time.perf_counter() - started
        environment._loaded_dynamic_motion_info = trajectory.get("dynamic_motion_info")
        replay_arguments["left_joint_path"] = trajectory["left_joint_path"]
        replay_arguments["right_joint_path"] = trajectory["right_joint_path"]
        environment.set_path_lst(replay_arguments)

        started = time.perf_counter()
        try:
            info = environment.play_once()
            _post_play_success_contract(task, environment)
        finally:
            replay_seconds = time.perf_counter() - started
        started = time.perf_counter()
        try:
            replay_success = bool(
                environment.plan_success and environment.check_success()
            )
        finally:
            success_check_seconds = time.perf_counter() - started
        if not replay_success:
            failure_kind = "replay_success_check_failure"
            failure_message = (
                "trajectory replay did not satisfy the unchanged success contract"
            )
            raise RuntimeError(failure_message)

        clear_frequency = int(base_arguments["clear_cache_freq"])
        _close_environment(
            environment,
            clear_cache=((claim.output_index + 1) % clear_frequency == 0),
        )
        environment_closed = True

        started = time.perf_counter()
        try:
            frame_count, hdf5_sha256 = _package_frames_to_hdf5(
                frames=environment._absolute_motion_frames,
                temporary_hdf5=temporary_hdf5,
            )
        finally:
            package_seconds = time.perf_counter() - started
        queue.begin_publish(
            claim=claim,
            replay_setup_seconds=replay_setup_seconds,
            replay_seconds=replay_seconds,
            success_check_seconds=success_check_seconds,
            package_seconds=package_seconds,
            frame_count=frame_count,
            hdf5_sha256=hdf5_sha256,
            info=info,
        )
        publishing_started = True

        atomic_copy(claim.trajectory_path, canonical_trajectory)
        final_hdf5.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary_hdf5, final_hdf5)
        descriptor = os.open(final_hdf5.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        queue.finalize_publish(
            claim=claim,
            hdf5_path=final_hdf5,
            trajectory_path=canonical_trajectory,
        )
        camera_take_picture_seconds = float(
            getattr(
                environment.cameras,
                "_absolute_motion_take_picture_seconds",
                0.0,
            )
        )
        camera_readback_seconds = float(
            getattr(
                environment.cameras,
                "_absolute_motion_readback_seconds",
                0.0,
            )
        )
        scene_update_seconds = float(
            getattr(
                environment,
                "_absolute_motion_scene_update_seconds",
                0.0,
            )
        )
        environment._absolute_motion_frames = []
        print(
            json.dumps(
                {
                    "event": "render_success",
                    "output_index": claim.output_index,
                    "seed": claim.seed,
                    "worker_id": claim.worker_id,
                    "frame_count": frame_count,
                    "replay_seconds": replay_seconds,
                    "package_seconds": package_seconds,
                    "scene_update_seconds": scene_update_seconds,
                    "camera_take_picture_seconds": (camera_take_picture_seconds),
                    "camera_readback_seconds": camera_readback_seconds,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return
    except Exception as error:
        if publishing_started:
            raise
        if failure_message != str(error):
            failure_kind = f"exception:{type(error).__name__}"
            failure_message = str(error)
        traceback.print_exc()
    finally:
        if not environment_closed:
            _close_environment(environment, clear_cache=True)
        if not publishing_started:
            temporary_hdf5.unlink(missing_ok=True)
        environment._absolute_motion_frames = []

    retryable_resource_failure = _is_retryable_render_resource_failure(failure_message)
    if retryable_resource_failure:
        queue_requeued = False
        try:
            queue.retry_render(
                claim=claim,
                replay_setup_seconds=replay_setup_seconds,
                replay_seconds=replay_seconds,
                success_check_seconds=success_check_seconds,
                failure_kind=failure_kind,
                failure_message=failure_message,
            )
            queue_requeued = True
        except Exception:
            traceback.print_exc()
    else:
        queue.reject_render(
            claim=claim,
            replay_setup_seconds=replay_setup_seconds,
            replay_seconds=replay_seconds,
            success_check_seconds=success_check_seconds,
            failure_kind=failure_kind,
            failure_message=failure_message,
        )
    print(
        json.dumps(
            {
                "event": (
                    "render_retry" if retryable_resource_failure else "render_failure"
                ),
                "output_index": claim.output_index,
                "seed": claim.seed,
                "worker_id": claim.worker_id,
                "failure_kind": failure_kind,
                "failure_message": failure_message,
                "queue_requeued": (
                    queue_requeued if retryable_resource_failure else None
                ),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if retryable_resource_failure:
        _exec_renderer(
            reason="resource_failure",
            attempts=1,
            worker_id=claim.worker_id,
        )


def _exec_renderer(
    *,
    reason: str,
    attempts: int,
    worker_id: str,
) -> None:
    print(
        json.dumps(
            {
                "event": "renderer_recycle",
                "reason": reason,
                "attempts": attempts,
                "worker_id": worker_id,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    os.execv(
        sys.executable,
        [
            sys.executable,
            "-m",
            "dynamicwam.training.domino_streaming_worker",
            *sys.argv[1:],
        ],
    )
    raise RuntimeError("Renderer exec unexpectedly returned")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--role",
        required=True,
        choices=("initialize", "planner", "renderer", "validate", "status"),
    )
    parser.add_argument("--domino-root", required=True)
    parser.add_argument("--job-root", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--task-config", required=True)
    parser.add_argument("--target-episodes", required=True, type=int)
    parser.add_argument("--worker-id", default=f"worker-{os.getpid()}")
    parser.add_argument("--ready-buffer-episodes", type=int, default=8)
    parser.add_argument("--max-new-attempts", type=int, default=20000)
    parser.add_argument("--poll-seconds", type=float, default=0.25)
    parser.add_argument("--renderer-recycle-attempts", required=True, type=int)
    return parser


def main() -> None:
    arguments = _build_parser().parse_args()
    domino_root = Path(arguments.domino_root).resolve()
    if not (domino_root / "envs" / "_base_task.py").is_file():
        raise FileNotFoundError(f"Invalid DOMINO root: {domino_root}")
    if arguments.target_episodes <= 0:
        raise ValueError("target-episodes must be positive")
    if arguments.ready_buffer_episodes < 0:
        raise ValueError("ready-buffer-episodes must be non-negative")
    if arguments.max_new_attempts <= 0 or arguments.poll_seconds <= 0:
        raise ValueError("Worker limits must be positive")
    if arguments.renderer_recycle_attempts <= 0:
        raise ValueError("Renderer recycle attempts must be positive")
    os.chdir(domino_root)
    sys.path.insert(0, str(domino_root))

    queue = _queue_from_arguments(arguments)
    if arguments.role == "initialize":
        print(
            json.dumps(queue.initialize().__dict__, sort_keys=True),
            flush=True,
        )
    elif arguments.role == "planner":
        run_planner(arguments)
    elif arguments.role == "renderer":
        run_renderer(arguments)
    elif arguments.role == "validate":
        queue.open_existing()
        print(
            json.dumps(queue.validate_complete(), sort_keys=True),
            flush=True,
        )
    elif arguments.role == "status":
        queue.open_existing()
        print(
            json.dumps(queue.status().__dict__, sort_keys=True),
            flush=True,
        )


if __name__ == "__main__":
    main()
