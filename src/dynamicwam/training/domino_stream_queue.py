"""Crash-recoverable ordered queue for exact-time DOMINO collection."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import time
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

STREAM_SCHEMA_VERSION = 1
STREAM_DIRECTORY_NAME = ".absolute_motion_stream_v2"


@dataclass(frozen=True)
class SeedClaim:
    seed: int
    worker_id: str


@dataclass(frozen=True)
class RenderClaim:
    item_id: int
    sequence_index: int
    seed: int
    trajectory_path: Path
    trajectory_sha256: str
    output_index: int
    worker_id: str


@dataclass(frozen=True)
class QueueStatus:
    target_episodes: int
    outputs: int
    ready: int
    rendering: int
    planning: int
    accepted: int
    rejected: int
    failed_attempts: int

    @property
    def complete(self) -> bool:
        return self.outputs == self.target_episodes


@dataclass(frozen=True)
class HDF5EpisodeInfo:
    frame_count: int
    sha256: str
    first_sim_step: int
    last_sim_step: int
    first_sim_time_seconds: float
    last_sim_time_seconds: float


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    try:
        with source.open("rb") as reader, temporary.open("wb") as writer:
            shutil.copyfileobj(reader, writer, length=1024 * 1024)
            writer.flush()
            os.fsync(writer.fileno())
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(path: Path, payload: Any) -> None:
    encoded = (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    atomic_write_bytes(path, encoded)


def numbered_files(
    directory: Path,
    *,
    prefix: str,
    suffix: str,
) -> list[int]:
    if not directory.is_dir():
        return []
    result: list[int] = []
    for path in directory.iterdir():
        name = path.name
        if not name.startswith(prefix) or not name.endswith(suffix):
            continue
        value = name[len(prefix) : len(name) - len(suffix)]
        if value.isdigit():
            result.append(int(value))
    return sorted(result)


def validate_hdf5_episode(path: Path) -> HDF5EpisodeInfo:
    import h5py
    import numpy as np

    from dynamicwam.training.data.robotwin2.aloha_qpos import (
        aloha_qpos_is_valid,
    )
    from dynamicwam.training.data.robotwin2.convert_exact_domino import (
        INTERCEPTION_FIELDS,
    )

    required_observations = (
        "observation/head_camera/rgb",
        "observation/left_camera/rgb",
        "observation/right_camera/rgb",
        "observation/front_camera/rgb",
        "joint_action/left_arm",
        "joint_action/right_arm",
        "joint_action/vector",
    )
    with h5py.File(path, "r") as handle:
        missing = [name for name in required_observations if name not in handle]
        if missing:
            raise RuntimeError(f"HDF5 episode is missing datasets: {missing}")
        if "interception" not in handle:
            raise RuntimeError(f"HDF5 episode has no interception group: {path}")
        interception = handle["interception"]
        if set(interception.keys()) != set(INTERCEPTION_FIELDS):
            raise RuntimeError(f"HDF5 interception schema differs from v2: {path}")

        frame_count = int(handle[required_observations[0]].shape[0])
        if frame_count <= 0:
            raise RuntimeError(f"HDF5 episode has no frames: {path}")
        for name in required_observations:
            if int(handle[name].shape[0]) != frame_count:
                raise RuntimeError(f"HDF5 dataset has a different frame count: {name}")
        for name in INTERCEPTION_FIELDS:
            dataset = interception[name]
            if dataset.ndim == 0 or int(dataset.shape[0]) != frame_count:
                raise RuntimeError(
                    "HDF5 interception field is not frame-synchronous: "
                    f"{name} in {path}"
                )
            if dataset.dtype.kind == "f" and not np.all(
                np.isfinite(np.asarray(dataset))
            ):
                raise RuntimeError(
                    f"HDF5 interception field is non-finite: {name} in {path}"
                )

        qpos = np.asarray(handle["joint_action/vector"])
        if qpos.shape != (frame_count, 14) or not aloha_qpos_is_valid(qpos):
            raise RuntimeError(f"Invalid qpos or pinned Aloha joint limits: {path}")

        schema = np.asarray(handle["interception/schema_version"])
        frame_index = np.asarray(handle["interception/frame_index"])
        sim_step = np.asarray(handle["interception/sim_step_index"])
        sim_time = np.asarray(handle["interception/sim_time_seconds"])
        timestep = np.asarray(handle["interception/sim_timestep_seconds"])
        if (
            schema.dtype.kind not in {"i", "u"}
            or not np.all(schema == 2)
            or not np.array_equal(
                frame_index,
                np.arange(frame_count, dtype=frame_index.dtype),
            )
        ):
            raise RuntimeError(f"Invalid interception schema: {path}")
        if (
            sim_step.dtype.kind not in {"i", "u"}
            or sim_time.dtype != np.dtype("float64")
            or timestep.dtype != np.dtype("float64")
            or not np.all(np.isfinite(sim_time))
            or not np.all(np.isfinite(timestep))
            or not np.all(timestep > 0)
            or not np.all(timestep == timestep[0])
        ):
            raise RuntimeError(f"Invalid exact simulator clock dtype: {path}")

        step_delta = np.diff(sim_step.astype(np.int64))
        time_delta = np.diff(sim_time)
        if np.any(step_delta < 0) or np.any(time_delta < 0):
            raise RuntimeError(f"Simulator clock moves backwards: {path}")
        if not np.array_equal(step_delta == 0, time_delta == 0):
            raise RuntimeError(f"Simulator step/time duplicate frames disagree: {path}")
        expected_delta = step_delta.astype(np.float64) * timestep[0]
        tolerance = max(
            1.0e-12,
            float(timestep[0]) * max(1, frame_count) * 1.0e-11,
        )
        if not np.allclose(
            time_delta,
            expected_delta,
            rtol=0.0,
            atol=tolerance,
        ):
            raise RuntimeError(f"Simulator step/time increments disagree: {path}")

    return HDF5EpisodeInfo(
        frame_count=frame_count,
        sha256=sha256_file(path),
        first_sim_step=int(sim_step[0]),
        last_sim_step=int(sim_step[-1]),
        first_sim_time_seconds=float(sim_time[0]),
        last_sim_time_seconds=float(sim_time[-1]),
    )


class StreamQueue:
    def __init__(
        self,
        *,
        job_root: Path,
        task: str,
        split: str,
        task_config: str,
        target_episodes: int,
    ):
        self.job_root = job_root.resolve()
        self.task = task
        self.split = split
        self.task_config = task_config
        self.target_episodes = int(target_episodes)
        self.stream_root = self.job_root / STREAM_DIRECTORY_NAME
        self.database_path = self.stream_root / "queue.sqlite3"
        self.item_root = self.stream_root / "trajectories"

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.database_path,
            timeout=60.0,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=60000")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _set_meta(
        connection: sqlite3.Connection,
        key: str,
        value: Any,
    ) -> None:
        connection.execute(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
            (key, json.dumps(value, sort_keys=True)),
        )

    @staticmethod
    def _get_meta(connection: sqlite3.Connection, key: str) -> Any:
        row = connection.execute(
            "SELECT value FROM metadata WHERE key = ?",
            (key,),
        ).fetchone()
        if row is None:
            raise RuntimeError(f"Streaming queue metadata is missing {key!r}")
        return json.loads(str(row["value"]))

    def _create_schema(self, connection: sqlite3.Connection) -> None:
        connection.executescript(
            """
            CREATE TABLE metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE attempts (
                seed INTEGER PRIMARY KEY,
                status TEXT NOT NULL,
                worker_id TEXT,
                started_at REAL,
                finished_at REAL,
                setup_seconds REAL,
                play_seconds REAL,
                success_check_seconds REAL,
                plan_success INTEGER,
                check_success INTEGER,
                failure_kind TEXT,
                failure_message TEXT,
                trajectory_path TEXT,
                trajectory_sha256 TEXT
            );
            CREATE TABLE items (
                item_id INTEGER PRIMARY KEY AUTOINCREMENT,
                sequence_index INTEGER UNIQUE NOT NULL,
                seed INTEGER UNIQUE NOT NULL,
                trajectory_path TEXT NOT NULL,
                trajectory_sha256 TEXT NOT NULL,
                source TEXT NOT NULL,
                status TEXT NOT NULL,
                worker_id TEXT,
                output_index INTEGER UNIQUE,
                render_started_at REAL,
                render_finished_at REAL,
                replay_setup_seconds REAL,
                replay_seconds REAL,
                success_check_seconds REAL,
                package_seconds REAL,
                frame_count INTEGER,
                hdf5_sha256 TEXT,
                info_json TEXT,
                failure_kind TEXT,
                failure_message TEXT
            );
            CREATE TABLE outputs (
                output_index INTEGER PRIMARY KEY,
                item_id INTEGER UNIQUE NOT NULL,
                seed INTEGER UNIQUE NOT NULL,
                hdf5_path TEXT NOT NULL,
                trajectory_path TEXT NOT NULL,
                frame_count INTEGER NOT NULL,
                hdf5_sha256 TEXT NOT NULL,
                created_at REAL NOT NULL,
                FOREIGN KEY(item_id) REFERENCES items(item_id)
            );
            """
        )

    def initialize(self) -> QueueStatus:
        self.job_root.mkdir(parents=True, exist_ok=True)
        self.stream_root.mkdir(parents=True, exist_ok=True)
        self.item_root.mkdir(parents=True, exist_ok=True)
        if self.database_path.exists():
            self._validate_identity()
            self.recover()
            return self.status()

        temporary_database = self.database_path.with_name(
            f".{self.database_path.name}.tmp.{os.getpid()}"
        )
        temporary_database.unlink(missing_ok=True)
        connection = sqlite3.connect(
            temporary_database,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        transaction_open = False
        try:
            connection.execute("PRAGMA journal_mode=DELETE")
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute("BEGIN IMMEDIATE")
            transaction_open = True
            self._create_schema(connection)
            self._initialize_empty(connection)
            connection.commit()
            transaction_open = False
            connection.close()
            os.replace(temporary_database, self.database_path)
            _fsync_directory(self.stream_root)
        except Exception:
            if transaction_open:
                connection.rollback()
            with suppress(sqlite3.ProgrammingError):
                connection.close()
            temporary_database.unlink(missing_ok=True)
            raise
        self._validate_identity()
        self.recover()
        return self.status()

    def open_existing(self) -> QueueStatus:
        if not self.database_path.is_file():
            raise FileNotFoundError(
                f"Streaming queue is not initialized: {self.database_path}"
            )
        self._validate_identity()
        return self.status()

    def _validate_identity(self) -> None:
        with self._transaction() as connection:
            expected = {
                "schema_version": STREAM_SCHEMA_VERSION,
                "task": self.task,
                "split": self.split,
                "task_config": self.task_config,
            }
            actual = {key: self._get_meta(connection, key) for key in expected}
            if actual != expected:
                raise RuntimeError(
                    f"Streaming queue identity differs: {actual} != {expected}"
                )
            actual_target = int(self._get_meta(connection, "target_episodes"))
            if actual_target != self.target_episodes:
                raise RuntimeError(
                    "Streaming queue target differs: "
                    f"{actual_target} != {self.target_episodes}"
                )

    def _initialize_empty(self, connection: sqlite3.Connection) -> None:
        candidates = (
            self.job_root / "seed.txt",
            self.job_root / "scene_info.json",
            self.job_root / "data",
            self.job_root / "_traj_data",
        )
        conflicting = [
            str(path)
            for path in candidates
            if path.is_file() or (path.is_dir() and any(path.iterdir()))
        ]
        if conflicting:
            raise RuntimeError(
                "Collection outputs exist without their streaming queue; "
                "move them out of the job root before starting a fresh run: "
                f"{conflicting}"
            )
        metadata = {
            "schema_version": STREAM_SCHEMA_VERSION,
            "task": self.task,
            "split": self.split,
            "task_config": self.task_config,
            "target_episodes": self.target_episodes,
            "next_candidate_seed": 0,
            "next_commit_seed": 0,
            "next_item_sequence": 0,
            "new_attempt_count": 0,
        }
        for key, value in metadata.items():
            self._set_meta(connection, key, value)

    def recover(self) -> None:
        with self._transaction() as connection:
            connection.execute(
                """
                UPDATE attempts
                SET status = 'pending', worker_id = NULL
                WHERE status = 'planning'
                """
            )
            rows = connection.execute(
                """
                SELECT item_id, status, output_index, trajectory_sha256
                FROM items
                WHERE status IN ('rendering', 'publishing')
                ORDER BY sequence_index
                """
            ).fetchall()
            for row in rows:
                item_id = int(row["item_id"])
                output_index = row["output_index"]
                final_hdf5 = (
                    self.job_root / "data" / f"episode{int(output_index)}.hdf5"
                    if output_index is not None
                    else None
                )
                canonical_trajectory = (
                    self.job_root / "_traj_data" / f"episode{int(output_index)}.pkl"
                    if output_index is not None
                    else None
                )
                if (
                    row["status"] == "publishing"
                    and final_hdf5 is not None
                    and final_hdf5.is_file()
                    and canonical_trajectory is not None
                    and canonical_trajectory.is_file()
                    and sha256_file(canonical_trajectory)
                    == str(row["trajectory_sha256"])
                ):
                    self._finalize_existing_publish(
                        connection,
                        item_id=item_id,
                        output_index=int(output_index),
                        hdf5_path=final_hdf5,
                        trajectory_path=canonical_trajectory,
                    )
                else:
                    if final_hdf5 is not None and final_hdf5.is_file():
                        raise RuntimeError(
                            "Cannot recover partially published HDF5 episode: "
                            f"{final_hdf5}"
                        )
                    connection.execute(
                        """
                        UPDATE items
                        SET status = 'ready', worker_id = NULL,
                            output_index = NULL
                        WHERE item_id = ?
                        """,
                        (item_id,),
                    )
            self._advance_commits(connection)
        self.rewrite_public_manifests()

    def _finalize_existing_publish(
        self,
        connection: sqlite3.Connection,
        *,
        item_id: int,
        output_index: int,
        hdf5_path: Path,
        trajectory_path: Path,
    ) -> None:
        item = connection.execute(
            "SELECT seed FROM items WHERE item_id = ?",
            (item_id,),
        ).fetchone()
        if item is None:
            raise RuntimeError(f"Unknown publishing item {item_id}")
        info = validate_hdf5_episode(hdf5_path)
        connection.execute(
            """
            INSERT OR IGNORE INTO outputs(
                output_index, item_id, seed, hdf5_path,
                trajectory_path, frame_count, hdf5_sha256, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                output_index,
                item_id,
                int(item["seed"]),
                str(hdf5_path),
                str(trajectory_path),
                info.frame_count,
                info.sha256,
                time.time(),
            ),
        )
        connection.execute(
            """
            UPDATE items
            SET status = 'done', worker_id = NULL,
                frame_count = ?, hdf5_sha256 = ?
            WHERE item_id = ?
            """,
            (info.frame_count, info.sha256, item_id),
        )

    def _advance_commits(self, connection: sqlite3.Connection) -> None:
        next_seed = int(self._get_meta(connection, "next_commit_seed"))
        next_sequence = int(self._get_meta(connection, "next_item_sequence"))
        while True:
            attempt = connection.execute(
                "SELECT * FROM attempts WHERE seed = ?",
                (next_seed,),
            ).fetchone()
            if attempt is None or attempt["status"] not in {
                "success",
                "failure",
            }:
                break
            if attempt["status"] == "success":
                trajectory_path = Path(str(attempt["trajectory_path"]))
                trajectory_sha256 = str(attempt["trajectory_sha256"])
                if (
                    not trajectory_path.is_file()
                    or sha256_file(trajectory_path) != trajectory_sha256
                ):
                    raise RuntimeError(
                        f"Successful seed has no valid trajectory payload: {next_seed}"
                    )
                connection.execute(
                    """
                    INSERT INTO items(
                        sequence_index, seed, trajectory_path,
                        trajectory_sha256, source, status
                    ) VALUES (?, ?, ?, ?, 'planned', 'ready')
                    """,
                    (
                        next_sequence,
                        next_seed,
                        str(trajectory_path),
                        trajectory_sha256,
                    ),
                )
                connection.execute(
                    "UPDATE attempts SET status = 'committed' WHERE seed = ?",
                    (next_seed,),
                )
                next_sequence += 1
            else:
                connection.execute(
                    """
                    UPDATE attempts
                    SET status = 'committed_failure'
                    WHERE seed = ?
                    """,
                    (next_seed,),
                )
            next_seed += 1
        self._set_meta(connection, "next_commit_seed", next_seed)
        self._set_meta(
            connection,
            "next_item_sequence",
            next_sequence,
        )

    def claim_seed(
        self,
        *,
        worker_id: str,
        ready_buffer_episodes: int,
        max_new_attempts: int,
    ) -> SeedClaim | None:
        with self._transaction() as connection:
            self._advance_commits(connection)
            outputs = int(
                connection.execute("SELECT COUNT(*) AS count FROM outputs").fetchone()[
                    "count"
                ]
            )
            if outputs >= self.target_episodes:
                return None
            unpublished = int(
                connection.execute(
                    """
                    SELECT COUNT(*) AS count
                    FROM items
                    WHERE status IN ('ready', 'rendering', 'publishing')
                    """
                ).fetchone()["count"]
            )
            unresolved = int(
                connection.execute(
                    """
                    SELECT COUNT(*) AS count
                    FROM attempts
                    WHERE status IN ('pending', 'planning', 'success')
                    """
                ).fetchone()["count"]
            )
            remaining = self.target_episodes - outputs
            planning_capacity = min(
                remaining,
                max(1, int(ready_buffer_episodes)),
            )
            if unpublished + unresolved >= planning_capacity:
                return None

            pending = connection.execute(
                """
                SELECT seed FROM attempts
                WHERE status = 'pending'
                ORDER BY seed
                LIMIT 1
                """
            ).fetchone()
            if pending is not None:
                seed = int(pending["seed"])
                connection.execute(
                    """
                    UPDATE attempts
                    SET status = 'planning', worker_id = ?, started_at = ?
                    WHERE seed = ?
                    """,
                    (worker_id, time.time(), seed),
                )
                return SeedClaim(seed=seed, worker_id=worker_id)

            new_attempt_count = int(self._get_meta(connection, "new_attempt_count"))
            if new_attempt_count >= int(max_new_attempts):
                raise RuntimeError(
                    "Streaming planner exhausted max_new_attempts="
                    f"{int(max_new_attempts)}"
                )
            seed = int(self._get_meta(connection, "next_candidate_seed"))
            self._set_meta(
                connection,
                "next_candidate_seed",
                seed + 1,
            )
            self._set_meta(
                connection,
                "new_attempt_count",
                new_attempt_count + 1,
            )
            connection.execute(
                """
                INSERT INTO attempts(
                    seed, status, worker_id, started_at
                ) VALUES (?, 'planning', ?, ?)
                """,
                (seed, worker_id, time.time()),
            )
            return SeedClaim(seed=seed, worker_id=worker_id)

    def complete_seed_success(
        self,
        *,
        claim: SeedClaim,
        trajectory_path: Path,
        trajectory_sha256: str,
        setup_seconds: float,
        play_seconds: float,
        success_check_seconds: float,
    ) -> None:
        with self._transaction() as connection:
            attempt = connection.execute(
                "SELECT status, worker_id FROM attempts WHERE seed = ?",
                (claim.seed,),
            ).fetchone()
            if (
                attempt is None
                or attempt["status"] != "planning"
                or attempt["worker_id"] != claim.worker_id
            ):
                raise RuntimeError(f"Seed claim ownership changed: {claim}")
            connection.execute(
                """
                UPDATE attempts
                SET status = 'success', finished_at = ?,
                    setup_seconds = ?, play_seconds = ?,
                    success_check_seconds = ?,
                    plan_success = 1, check_success = 1,
                    trajectory_path = ?, trajectory_sha256 = ?
                WHERE seed = ?
                """,
                (
                    time.time(),
                    setup_seconds,
                    play_seconds,
                    success_check_seconds,
                    str(trajectory_path),
                    trajectory_sha256,
                    claim.seed,
                ),
            )
            self._advance_commits(connection)

    def complete_seed_failure(
        self,
        *,
        claim: SeedClaim,
        setup_seconds: float,
        play_seconds: float,
        success_check_seconds: float,
        plan_success: bool,
        check_success: bool,
        failure_kind: str,
        failure_message: str,
    ) -> None:
        with self._transaction() as connection:
            attempt = connection.execute(
                "SELECT status, worker_id FROM attempts WHERE seed = ?",
                (claim.seed,),
            ).fetchone()
            if (
                attempt is None
                or attempt["status"] != "planning"
                or attempt["worker_id"] != claim.worker_id
            ):
                raise RuntimeError(f"Seed claim ownership changed: {claim}")
            connection.execute(
                """
                UPDATE attempts
                SET status = 'failure', finished_at = ?,
                    setup_seconds = ?, play_seconds = ?,
                    success_check_seconds = ?,
                    plan_success = ?, check_success = ?,
                    failure_kind = ?, failure_message = ?
                WHERE seed = ?
                """,
                (
                    time.time(),
                    setup_seconds,
                    play_seconds,
                    success_check_seconds,
                    int(plan_success),
                    int(check_success),
                    failure_kind,
                    failure_message,
                    claim.seed,
                ),
            )
            self._advance_commits(connection)

    def claim_render(self, *, worker_id: str) -> RenderClaim | None:
        with self._transaction() as connection:
            outputs = int(
                connection.execute("SELECT COUNT(*) AS count FROM outputs").fetchone()[
                    "count"
                ]
            )
            if outputs >= self.target_episodes:
                return None
            item = connection.execute(
                """
                SELECT * FROM items
                WHERE status = 'ready'
                ORDER BY sequence_index
                LIMIT 1
                """
            ).fetchone()
            if item is None:
                return None
            connection.execute(
                """
                UPDATE items
                SET status = 'rendering', worker_id = ?,
                    output_index = ?, render_started_at = ?,
                    render_finished_at = NULL,
                    replay_setup_seconds = NULL,
                    replay_seconds = NULL,
                    success_check_seconds = NULL,
                    package_seconds = NULL,
                    frame_count = NULL,
                    hdf5_sha256 = NULL,
                    info_json = NULL,
                    failure_kind = NULL,
                    failure_message = NULL
                WHERE item_id = ?
                """,
                (
                    worker_id,
                    outputs,
                    time.time(),
                    int(item["item_id"]),
                ),
            )
            return RenderClaim(
                item_id=int(item["item_id"]),
                sequence_index=int(item["sequence_index"]),
                seed=int(item["seed"]),
                trajectory_path=Path(str(item["trajectory_path"])),
                trajectory_sha256=str(item["trajectory_sha256"]),
                output_index=outputs,
                worker_id=worker_id,
            )

    def reject_render(
        self,
        *,
        claim: RenderClaim,
        replay_setup_seconds: float,
        replay_seconds: float,
        success_check_seconds: float,
        failure_kind: str,
        failure_message: str,
    ) -> None:
        with self._transaction() as connection:
            self._assert_render_claim(connection, claim, "rendering")
            connection.execute(
                """
                UPDATE items
                SET status = 'rejected', worker_id = NULL,
                    output_index = NULL, render_finished_at = ?,
                    replay_setup_seconds = ?, replay_seconds = ?,
                    success_check_seconds = ?,
                    failure_kind = ?, failure_message = ?
                WHERE item_id = ?
                """,
                (
                    time.time(),
                    replay_setup_seconds,
                    replay_seconds,
                    success_check_seconds,
                    failure_kind,
                    failure_message,
                    claim.item_id,
                ),
            )

    def retry_render(
        self,
        *,
        claim: RenderClaim,
        replay_setup_seconds: float,
        replay_seconds: float,
        success_check_seconds: float,
        failure_kind: str,
        failure_message: str,
    ) -> None:
        with self._transaction() as connection:
            self._assert_render_claim(connection, claim, "rendering")
            connection.execute(
                """
                UPDATE items
                SET status = 'ready', worker_id = NULL,
                    output_index = NULL, render_finished_at = ?,
                    replay_setup_seconds = ?, replay_seconds = ?,
                    success_check_seconds = ?,
                    failure_kind = ?, failure_message = ?
                WHERE item_id = ?
                """,
                (
                    time.time(),
                    replay_setup_seconds,
                    replay_seconds,
                    success_check_seconds,
                    failure_kind,
                    failure_message,
                    claim.item_id,
                ),
            )

    def recover_renderer_claims(self, *, worker_id: str) -> int:
        with self._transaction() as connection:
            publishing = int(
                connection.execute(
                    """
                    SELECT COUNT(*) AS count
                    FROM items
                    WHERE status = 'publishing' AND worker_id = ?
                    """,
                    (worker_id,),
                ).fetchone()["count"]
            )
            if publishing:
                raise RuntimeError(
                    "Renderer process recovery found an in-flight publish: "
                    f"worker_id={worker_id!r}, count={publishing}"
                )
            rendering = int(
                connection.execute(
                    """
                    SELECT COUNT(*) AS count
                    FROM items
                    WHERE status = 'rendering' AND worker_id = ?
                    """,
                    (worker_id,),
                ).fetchone()["count"]
            )
            if rendering > 1:
                raise RuntimeError(
                    "Renderer owns more than one render claim: "
                    f"worker_id={worker_id!r}, count={rendering}"
                )
            connection.execute(
                """
                UPDATE items
                SET status = 'ready', worker_id = NULL,
                    output_index = NULL
                WHERE status = 'rendering' AND worker_id = ?
                """,
                (worker_id,),
            )
        return rendering

    @staticmethod
    def _assert_render_claim(
        connection: sqlite3.Connection,
        claim: RenderClaim,
        expected_status: str,
    ) -> sqlite3.Row:
        item = connection.execute(
            "SELECT * FROM items WHERE item_id = ?",
            (claim.item_id,),
        ).fetchone()
        if (
            item is None
            or item["status"] != expected_status
            or item["worker_id"] != claim.worker_id
            or int(item["output_index"]) != claim.output_index
        ):
            raise RuntimeError(f"Render claim ownership changed: {claim}")
        return item

    def begin_publish(
        self,
        *,
        claim: RenderClaim,
        replay_setup_seconds: float,
        replay_seconds: float,
        success_check_seconds: float,
        package_seconds: float,
        frame_count: int,
        hdf5_sha256: str,
        info: Any,
    ) -> None:
        with self._transaction() as connection:
            self._assert_render_claim(connection, claim, "rendering")
            connection.execute(
                """
                UPDATE items
                SET status = 'publishing', render_finished_at = ?,
                    replay_setup_seconds = ?, replay_seconds = ?,
                    success_check_seconds = ?, package_seconds = ?,
                    frame_count = ?, hdf5_sha256 = ?, info_json = ?
                WHERE item_id = ?
                """,
                (
                    time.time(),
                    replay_setup_seconds,
                    replay_seconds,
                    success_check_seconds,
                    package_seconds,
                    frame_count,
                    hdf5_sha256,
                    json.dumps(info, ensure_ascii=False, sort_keys=True),
                    claim.item_id,
                ),
            )

    def finalize_publish(
        self,
        *,
        claim: RenderClaim,
        hdf5_path: Path,
        trajectory_path: Path,
    ) -> None:
        with self._transaction() as connection:
            item = self._assert_render_claim(
                connection,
                claim,
                "publishing",
            )
            if (
                not hdf5_path.is_file()
                or not trajectory_path.is_file()
                or sha256_file(trajectory_path) != claim.trajectory_sha256
            ):
                raise RuntimeError(
                    f"Published files are incomplete for output {claim.output_index}"
                )
            info = validate_hdf5_episode(hdf5_path)
            if info.frame_count != int(item["frame_count"]) or info.sha256 != str(
                item["hdf5_sha256"]
            ):
                raise RuntimeError(
                    f"Published HDF5 changed for output {claim.output_index}"
                )
            connection.execute(
                """
                INSERT INTO outputs(
                    output_index, item_id, seed, hdf5_path,
                    trajectory_path, frame_count, hdf5_sha256, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    claim.output_index,
                    claim.item_id,
                    claim.seed,
                    str(hdf5_path),
                    str(trajectory_path),
                    info.frame_count,
                    info.sha256,
                    time.time(),
                ),
            )
            connection.execute(
                """
                UPDATE items
                SET status = 'done', worker_id = NULL
                WHERE item_id = ?
                """,
                (claim.item_id,),
            )
        self.rewrite_public_manifests()

    def rewrite_public_manifests(self) -> None:
        if not self.database_path.is_file():
            return
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT outputs.output_index, outputs.seed, items.info_json
                FROM outputs
                JOIN items ON items.item_id = outputs.item_id
                ORDER BY outputs.output_index
                """
            ).fetchall()
        expected = list(range(len(rows)))
        actual = [int(row["output_index"]) for row in rows]
        if actual != expected:
            raise RuntimeError(f"Output database is not contiguous: {actual}")
        seeds = [int(row["seed"]) for row in rows]
        seed_text = "".join(f"{seed} " for seed in seeds).encode("utf-8")
        atomic_write_bytes(self.job_root / "seed.txt", seed_text)

        info_payload: dict[str, Any] = {}
        for row in rows:
            if row["info_json"] is not None:
                info_payload[f"episode_{int(row['output_index'])}"] = json.loads(
                    str(row["info_json"])
                )
        atomic_write_json(self.job_root / "scene_info.json", info_payload)

    def status(self) -> QueueStatus:
        with self._connect() as connection:
            outputs = int(
                connection.execute("SELECT COUNT(*) AS count FROM outputs").fetchone()[
                    "count"
                ]
            )
            counts = {
                status: int(count)
                for status, count in connection.execute(
                    """
                    SELECT status, COUNT(*) FROM items GROUP BY status
                    """
                ).fetchall()
            }
            planning = int(
                connection.execute(
                    """
                    SELECT COUNT(*) AS count FROM attempts
                    WHERE status IN ('pending', 'planning', 'success')
                    """
                ).fetchone()["count"]
            )
            failed_attempts = int(
                connection.execute(
                    """
                    SELECT COUNT(*) AS count FROM attempts
                    WHERE status = 'committed_failure'
                    """
                ).fetchone()["count"]
            )
        return QueueStatus(
            target_episodes=self.target_episodes,
            outputs=outputs,
            ready=counts.get("ready", 0),
            rendering=counts.get("rendering", 0) + counts.get("publishing", 0),
            planning=planning,
            accepted=sum(
                counts.get(value, 0)
                for value in ("ready", "rendering", "publishing", "done")
            ),
            rejected=counts.get("rejected", 0),
            failed_attempts=failed_attempts,
        )

    def validate_complete(self) -> dict[str, Any]:
        status = self.status()
        if not status.complete:
            raise RuntimeError(f"Streaming queue is incomplete: {status}")
        self._prune_completed_payloads()
        self._canonicalize_completed_trajectories()
        status = self.status()
        hdf5_ids = numbered_files(
            self.job_root / "data",
            prefix="episode",
            suffix=".hdf5",
        )
        trajectory_ids = numbered_files(
            self.job_root / "_traj_data",
            prefix="episode",
            suffix=".pkl",
        )
        expected = list(range(self.target_episodes))
        if hdf5_ids != expected or trajectory_ids != expected:
            raise RuntimeError(
                "Completed streaming output is not contiguous: "
                f"{len(hdf5_ids)} HDF5, {len(trajectory_ids)} trajectories"
            )
        seeds = [
            int(value) for value in (self.job_root / "seed.txt").read_text().split()
        ]
        if len(seeds) != self.target_episodes or len(seeds) != len(set(seeds)):
            raise RuntimeError(
                f"Completed streaming output has invalid seeds: {len(seeds)}"
            )

        total_frames = 0
        hdf5_digest = hashlib.sha256()
        with self._connect() as connection:
            outputs = connection.execute(
                "SELECT * FROM outputs ORDER BY output_index"
            ).fetchall()
        if len(outputs) != self.target_episodes:
            raise RuntimeError("Output database count differs from target")
        for output in outputs:
            output_index = int(output["output_index"])
            if int(output["seed"]) != seeds[output_index]:
                raise RuntimeError(f"Seed database mismatch at output {output_index}")
            hdf5_path = Path(str(output["hdf5_path"]))
            trajectory_path = Path(str(output["trajectory_path"]))
            info = validate_hdf5_episode(hdf5_path)
            if (
                info.sha256 != str(output["hdf5_sha256"])
                or info.frame_count != int(output["frame_count"])
                or not trajectory_path.is_file()
            ):
                raise RuntimeError(
                    f"Output database payload mismatch at {output_index}"
                )
            total_frames += info.frame_count
            hdf5_digest.update(bytes.fromhex(info.sha256))

        return {
            "task": self.task,
            "split": self.split,
            "episodes": self.target_episodes,
            "total_frames": total_frames,
            "seed_count": len(seeds),
            "hdf5_set_sha256": hdf5_digest.hexdigest(),
            "queue_status": status.__dict__,
        }

    def _canonicalize_completed_trajectories(self) -> None:
        stage = self.stream_root / "canonical_trajectories"
        shutil.rmtree(stage, ignore_errors=True)
        stage.mkdir(parents=True, exist_ok=False)
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT outputs.output_index, outputs.trajectory_path,
                       items.item_id, items.trajectory_sha256
                FROM outputs
                JOIN items ON items.item_id = outputs.item_id
                ORDER BY outputs.output_index
                """
            ).fetchall()
        expected = list(range(self.target_episodes))
        output_ids = [int(row["output_index"]) for row in rows]
        if output_ids != expected:
            raise RuntimeError(
                f"Cannot canonicalize a non-contiguous output set: {output_ids}"
            )

        try:
            for row in rows:
                output_index = int(row["output_index"])
                source = Path(str(row["trajectory_path"]))
                expected_sha256 = str(row["trajectory_sha256"])
                if not source.is_file() or sha256_file(source) != expected_sha256:
                    raise RuntimeError(
                        "Completed output has no valid trajectory payload: "
                        f"{output_index}"
                    )
                staged = stage / f"episode{output_index}.pkl"
                atomic_copy(source, staged)
                if sha256_file(staged) != expected_sha256:
                    raise RuntimeError(
                        "Staged trajectory changed while canonicalizing: "
                        f"{output_index}"
                    )

            canonical_root = self.job_root / "_traj_data"
            canonical_root.mkdir(parents=True, exist_ok=True)
            canonical_paths: dict[int, Path] = {}
            for row in rows:
                output_index = int(row["output_index"])
                canonical = canonical_root / f"episode{output_index}.pkl"
                atomic_copy(
                    stage / f"episode{output_index}.pkl",
                    canonical,
                )
                if sha256_file(canonical) != str(row["trajectory_sha256"]):
                    raise RuntimeError(
                        f"Canonical trajectory failed hash verification: {output_index}"
                    )
                canonical_paths[output_index] = canonical

            with self._transaction() as connection:
                for row in rows:
                    output_index = int(row["output_index"])
                    canonical = str(canonical_paths[output_index])
                    connection.execute(
                        """
                        UPDATE outputs
                        SET trajectory_path = ?
                        WHERE output_index = ?
                        """,
                        (canonical, output_index),
                    )
                    connection.execute(
                        """
                        UPDATE items
                        SET trajectory_path = ?
                        WHERE item_id = ?
                        """,
                        (canonical, int(row["item_id"])),
                    )

            for trajectory_id in numbered_files(
                canonical_root,
                prefix="episode",
                suffix=".pkl",
            ):
                if trajectory_id >= self.target_episodes:
                    (canonical_root / f"episode{trajectory_id}.pkl").unlink()
            for payload in self.item_root.glob("*.pkl"):
                payload.unlink()
        finally:
            shutil.rmtree(stage, ignore_errors=True)

    def _prune_completed_payloads(self) -> None:
        delete_paths: set[Path] = set()
        with self._transaction() as connection:
            unfinished_attempts = int(
                connection.execute(
                    """
                    SELECT COUNT(*) AS count
                    FROM attempts
                    WHERE status IN ('pending', 'planning', 'success')
                    """
                ).fetchone()["count"]
            )
            active_renders = int(
                connection.execute(
                    """
                    SELECT COUNT(*) AS count
                    FROM items
                    WHERE status IN ('rendering', 'publishing')
                    """
                ).fetchone()["count"]
            )
            if unfinished_attempts or active_renders:
                raise RuntimeError(
                    "Cannot finalize while planner or renderer work is active"
                )
            rows = connection.execute(
                """
                SELECT items.item_id, items.status, items.trajectory_path,
                       outputs.trajectory_path AS output_trajectory_path
                FROM items
                LEFT JOIN outputs ON outputs.item_id = items.item_id
                """
            ).fetchall()
            for row in rows:
                source = Path(str(row["trajectory_path"]))
                if row["status"] == "done":
                    output_path = Path(str(row["output_trajectory_path"]))
                    if source != output_path:
                        delete_paths.add(source)
                        connection.execute(
                            """
                            UPDATE items
                            SET trajectory_path = ?
                            WHERE item_id = ?
                            """,
                            (str(output_path), int(row["item_id"])),
                        )
                elif row["status"] in {"ready", "rejected", "unused"}:
                    delete_paths.add(source)
                    connection.execute(
                        """
                        UPDATE items
                        SET status = 'unused'
                        WHERE item_id = ?
                        """,
                        (int(row["item_id"]),),
                    )
        for path in delete_paths:
            path.unlink(missing_ok=True)
