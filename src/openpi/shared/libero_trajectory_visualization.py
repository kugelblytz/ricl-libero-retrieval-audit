from __future__ import annotations

import dataclasses
import json
import logging
import pathlib
import threading
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np

logger = logging.getLogger(__name__)


_MARKER_PREFIX = "openpi_traj_marker"
_MARKER_HIDDEN_POS = np.asarray([0.0, 0.0, -10.0], dtype=np.float32)
_CAMERA_PREFIX = "openpi_traj"
_POLICY_COLOR = (66, 170, 255)
_POLICY_HEAD_COLOR = (58, 225, 120)
_GENERATED_COLOR = (255, 70, 70)
_GENERATED_HEAD_COLOR = (255, 245, 110)
_QUERY_COLOR = (255, 255, 255)
_RETRIEVAL_COLORS = (
    (255, 145, 35),
    (255, 214, 76),
    (235, 92, 181),
    (143, 119, 255),
    (70, 220, 220),
)


@dataclasses.dataclass
class StepPose:
    timestep: int
    phase: str
    query_idx: int | None
    eef_pos: list[float]


@dataclasses.dataclass
class RetrievalTrajectory:
    query_idx: int
    timestep: int
    rank: int
    distance: float | None
    episode_path: str
    step_idx: int
    points: list[list[float]]


@dataclasses.dataclass
class GeneratedTrajectory:
    query_idx: int
    timestep: int
    action_scale: float
    points: list[list[float]]


class LiberoTrajectoryVisualizer:
    """Draws RICL retrieved and executed end-effector trajectories on LIBERO rollout frames.

    The renderer stores all raw 3D points in a sidecar JSON. Frame annotation uses a robust 2D
    projection fallback because offscreen robosuite marker APIs differ across LIBERO installs.
    """

    def __init__(
        self,
        *,
        horizon: int,
        top_k: int,
        frame_size: int,
        renderer: str = "mujoco",
        camera_name: str = "agentview",
        generated_action_scale: float = 1.0,
        workspace_margin_px: int = 16,
    ):
        if horizon <= 0:
            raise ValueError(f"trajectory horizon must be positive, got {horizon}.")
        if top_k <= 0:
            raise ValueError(f"trajectory top_k must be positive, got {top_k}.")
        self.horizon = horizon
        self.top_k = top_k
        self.frame_size = frame_size
        self.renderer = renderer
        self.camera_name = camera_name
        self.camera_names = _parse_camera_names(camera_name)
        self.generated_action_scale = generated_action_scale
        self.workspace_margin_px = workspace_margin_px
        self.step_poses: list[StepPose] = []
        self.retrieval_trajectories: list[RetrievalTrajectory] = []
        self.generated_trajectories: list[GeneratedTrajectory] = []
        self._active_retrievals: list[RetrievalTrajectory] = []
        self._active_generated: GeneratedTrajectory | None = None
        self._projection_bounds: tuple[np.ndarray, np.ndarray] | None = None
        self._scene_renderer: MuJoCoTrajectorySceneRenderer | None = None
        self._renderer_mode = renderer

    @property
    def renderer_mode(self) -> str:
        return self._renderer_mode

    def detect_renderer_mode(self, env: Any) -> str:
        """Record whether the environment appears to expose marker hooks.

        MuJoCo scene rendering is preferred because it adds real geoms to the rendered scene.
        The OpenCV projection remains available as a fallback for environments without an
        accessible MuJoCo model/data pair.
        """
        if self.renderer == "mujoco":
            try:
                self._scene_renderer = MuJoCoModelMarkerTrajectoryRenderer(
                    env,
                    width=self.frame_size,
                    height=self.frame_size,
                    camera_name=self.camera_name,
                )
                self._renderer_mode = "mujoco_model_marker_geoms"
                return self._renderer_mode
            except Exception as exc:
                logger.warning("Falling back to OpenCV trajectory overlay; MuJoCo scene renderer unavailable: %s", exc)

        self._scene_renderer = None
        self._renderer_mode = "opencv_2d_projection"
        return self._renderer_mode

    def record_step(self, *, timestep: int, obs: dict[str, Any], phase: str, query_idx: int | None) -> None:
        eef_pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float32)
        if eef_pos.shape != (3,):
            raise ValueError(f"Expected robot0_eef_pos with shape (3,), got {eef_pos.shape}.")
        self.step_poses.append(
            StepPose(
                timestep=int(timestep),
                phase=str(phase),
                query_idx=None if query_idx is None else int(query_idx),
                eef_pos=eef_pos.astype(float).tolist(),
            )
        )
        self._projection_bounds = None

    def update_retrievals(
        self,
        *,
        query_idx: int,
        timestep: int,
        retrieval_trace: list[dict[str, Any]],
    ) -> None:
        trajectories = []
        for item in retrieval_trace[: self.top_k]:
            trajectory = load_retrieval_trajectory(
                item,
                query_idx=query_idx,
                timestep=timestep,
                horizon=self.horizon,
            )
            trajectories.append(trajectory)
        self._active_retrievals = trajectories
        self.retrieval_trajectories.extend(trajectories)
        self._projection_bounds = None

    def update_generated_actions(
        self,
        *,
        query_idx: int,
        timestep: int,
        query_state: np.ndarray,
        action_chunk: np.ndarray,
    ) -> None:
        points = build_generated_action_trajectory(
            query_state,
            action_chunk,
            horizon=self.horizon,
            action_scale=self.generated_action_scale,
        )
        trajectory = GeneratedTrajectory(
            query_idx=int(query_idx),
            timestep=int(timestep),
            action_scale=float(self.generated_action_scale),
            points=points.astype(float).tolist(),
        )
        self._active_generated = trajectory
        self.generated_trajectories.append(trajectory)
        self._projection_bounds = None

    def render(self, frame: np.ndarray) -> np.ndarray:
        frame = np.asarray(frame).copy()
        if frame.ndim != 3 or frame.shape[-1] != 3:
            raise ValueError(f"Expected RGB frame with shape (H, W, 3), got {frame.shape}.")
        if not self.step_poses and not self._active_retrievals:
            return frame

        try:
            import cv2
        except ImportError:
            logger.warning("opencv-python is not available; trajectory video will be unannotated.")
            return frame

        policy_points = np.asarray([pose.eef_pos for pose in self.step_poses], dtype=np.float32)
        if len(policy_points) >= 2:
            _draw_polyline(
                frame,
                _project_points(policy_points, self._get_projection_bounds(), frame.shape, self.workspace_margin_px),
                _POLICY_COLOR,
                thickness=3,
                alpha=0.82,
                cv2=cv2,
            )
            _draw_circle(
                frame,
                _project_points(policy_points[-1:], self._get_projection_bounds(), frame.shape, self.workspace_margin_px)[0],
                radius=5,
                color=_POLICY_HEAD_COLOR,
                alpha=0.95,
                cv2=cv2,
            )

        for trajectory in self._active_retrievals:
            points = np.asarray(trajectory.points, dtype=np.float32)
            if len(points) == 0:
                continue
            color = _color_for_rank(trajectory.rank)
            alpha = 0.88 if trajectory.rank == 0 else 0.58
            projected = _project_points(points, self._get_projection_bounds(), frame.shape, self.workspace_margin_px)
            _draw_polyline(frame, projected, color, thickness=2, alpha=alpha, cv2=cv2)
            _draw_circle(frame, projected[0], radius=4, color=color, alpha=0.92, cv2=cv2)

        if self._active_generated is not None:
            generated_points = np.asarray(self._active_generated.points, dtype=np.float32)
            if len(generated_points) >= 2:
                projected_generated = _project_points(
                    generated_points,
                    self._get_projection_bounds(),
                    frame.shape,
                    self.workspace_margin_px,
                )
                _draw_polyline(frame, projected_generated, _GENERATED_COLOR, thickness=3, alpha=0.9, cv2=cv2)
                _draw_circle(frame, projected_generated[-1], radius=5, color=_GENERATED_HEAD_COLOR, alpha=0.95, cv2=cv2)

        if self.step_poses:
            query_point = np.asarray([self.step_poses[-1].eef_pos], dtype=np.float32)
            projected_query = _project_points(query_point, self._get_projection_bounds(), frame.shape, self.workspace_margin_px)[
                0
            ]
            _draw_circle(frame, projected_query, radius=3, color=_QUERY_COLOR, alpha=0.95, cv2=cv2)

        _draw_legend(frame, self._active_retrievals, cv2=cv2)
        return frame

    def render_scene(self, env: Any, fallback_frame: np.ndarray) -> np.ndarray:
        if self._scene_renderer is None:
            return self.render(fallback_frame)
        return self._scene_renderer.render(
            env,
            policy_points=np.asarray([pose.eef_pos for pose in self.step_poses], dtype=np.float32),
            retrievals=self._active_retrievals,
            generated=self._active_generated,
            fallback_frame=fallback_frame,
        )

    def close(self) -> None:
        if self._scene_renderer is not None:
            self._scene_renderer.close()
            self._scene_renderer = None

    def write_sidecar(self, path: pathlib.Path, *, final_video_path: str | pathlib.Path, trajectory_video_path: str | pathlib.Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "renderer_mode": self.renderer_mode,
            "requested_renderer": self.renderer,
            "camera_name": self.camera_name,
            "camera_names": self.camera_names,
            "trajectory_horizon": self.horizon,
            "trajectory_top_k": self.top_k,
            "generated_action_scale": self.generated_action_scale,
            "final_video_path": str(final_video_path),
            "trajectory_video_path": str(trajectory_video_path),
            "step_poses": [dataclasses.asdict(pose) for pose in self.step_poses],
            "retrieval_trajectories": [dataclasses.asdict(trajectory) for trajectory in self.retrieval_trajectories],
            "generated_trajectories": [dataclasses.asdict(trajectory) for trajectory in self.generated_trajectories],
        }
        path.write_text(json.dumps(payload, indent=2) + "\n")

    def _get_projection_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        if self._projection_bounds is not None:
            return self._projection_bounds

        chunks = []
        if self.step_poses:
            chunks.append(np.asarray([pose.eef_pos for pose in self.step_poses], dtype=np.float32))
        chunks.extend(np.asarray(trajectory.points, dtype=np.float32) for trajectory in self._active_retrievals if trajectory.points)
        if self._active_generated is not None and self._active_generated.points:
            chunks.append(np.asarray(self._active_generated.points, dtype=np.float32))
        if not chunks:
            bounds = (np.asarray([-0.35, -0.35, 0.7], dtype=np.float32), np.asarray([0.35, 0.45, 1.4], dtype=np.float32))
        else:
            points = np.concatenate(chunks, axis=0)
            low = np.nanmin(points, axis=0)
            high = np.nanmax(points, axis=0)
            span = np.maximum(high - low, 0.05)
            bounds = (low - 0.1 * span, high + 0.1 * span)
        self._projection_bounds = bounds
        return bounds


def load_retrieval_trajectory(
    retrieval_item: dict[str, Any],
    *,
    query_idx: int,
    timestep: int,
    horizon: int,
) -> RetrievalTrajectory:
    episode_path = pathlib.Path(str(retrieval_item["episode_path"]))
    step_idx = int(retrieval_item["step_idx"])
    with np.load(episode_path, allow_pickle=False) as episode:
        if "state" not in episode:
            raise KeyError(f"{episode_path} is missing required 'state' array.")
        states = np.asarray(episode["state"], dtype=np.float32)
    points = slice_trajectory_points(states, step_idx=step_idx, horizon=horizon)
    return RetrievalTrajectory(
        query_idx=int(query_idx),
        timestep=int(timestep),
        rank=int(retrieval_item.get("rank", 0)),
        distance=None if retrieval_item.get("distance") is None else float(retrieval_item["distance"]),
        episode_path=str(episode_path),
        step_idx=step_idx,
        points=points.astype(float).tolist(),
    )


def slice_trajectory_points(states: np.ndarray, *, step_idx: int, horizon: int) -> np.ndarray:
    states = np.asarray(states, dtype=np.float32)
    if states.ndim != 2 or states.shape[1] < 3:
        raise ValueError(f"Expected states with shape (T, >=3), got {states.shape}.")
    if len(states) == 0:
        raise ValueError("Cannot slice a trajectory from an empty state array.")
    if horizon <= 0:
        raise ValueError(f"horizon must be positive, got {horizon}.")
    step_idx = min(max(int(step_idx), 0), len(states) - 1)
    end = min(step_idx + horizon, len(states))
    return states[step_idx:end, :3]


def build_generated_action_trajectory(
    query_state: np.ndarray,
    action_chunk: np.ndarray,
    *,
    horizon: int,
    action_scale: float = 1.0,
) -> np.ndarray:
    query_state = np.asarray(query_state, dtype=np.float32)
    action_chunk = np.asarray(action_chunk, dtype=np.float32)
    if query_state.ndim != 1 or query_state.shape[0] < 3:
        raise ValueError(f"Expected query_state with shape (>=3,), got {query_state.shape}.")
    if action_chunk.ndim != 2 or action_chunk.shape[1] < 3:
        raise ValueError(f"Expected action_chunk with shape (T, >=3), got {action_chunk.shape}.")
    if horizon <= 0:
        raise ValueError(f"horizon must be positive, got {horizon}.")
    if len(action_chunk) == 0:
        return query_state[:3].reshape(1, 3).astype(np.float32)

    deltas = action_chunk[:horizon, :3] * float(action_scale)
    points = query_state[:3].reshape(1, 3) + np.cumsum(deltas, axis=0)
    return np.concatenate([query_state[:3].reshape(1, 3), points], axis=0).astype(np.float32)


def color_for_rank(rank: int) -> tuple[int, int, int]:
    return _color_for_rank(rank)


def install_mujoco_marker_xml_processor(env: Any, *, marker_count: int) -> None:
    if marker_count <= 0:
        raise ValueError(f"marker_count must be positive, got {marker_count}.")
    target = _find_xml_processor_target(env)
    if target is None:
        raise RuntimeError("Could not find a robosuite environment with set_xml_processor(...).")

    existing_processor = getattr(target, "_xml_processor", None)

    def processor(xml: str) -> str:
        if callable(existing_processor):
            xml = existing_processor(xml)
        return add_mujoco_marker_geoms_to_xml(xml, marker_count=marker_count)

    target.set_xml_processor(processor)


def add_mujoco_marker_geoms_to_xml(xml: str, *, marker_count: int) -> str:
    root = ET.fromstring(xml)
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("MuJoCo XML is missing a worldbody element.")
    _add_trajectory_cameras(worldbody)
    for existing in list(worldbody.findall("body")):
        name = existing.get("name", "")
        if name.startswith(f"{_MARKER_PREFIX}_body_"):
            worldbody.remove(existing)

    for marker_idx in range(marker_count):
        body = ET.SubElement(
            worldbody,
            "body",
            {
                "name": f"{_MARKER_PREFIX}_body_{marker_idx:04d}",
                "mocap": "true",
                "pos": _array_to_mjcf(_MARKER_HIDDEN_POS),
            },
        )
        ET.SubElement(
            body,
            "geom",
            {
                "name": f"{_MARKER_PREFIX}_geom_{marker_idx:04d}",
                "type": "sphere",
                "size": "0.01",
                "rgba": "0 0 0 0",
                "contype": "0",
                "conaffinity": "0",
                "group": "1",
            },
        )
    return ET.tostring(root, encoding="unicode")


class MuJoCoModelMarkerTrajectoryRenderer:
    def __init__(self, env: Any, *, width: int, height: int, camera_name: str):
        self.width = width
        self.height = height
        self.camera_names = _parse_camera_names(camera_name)
        self.sim = _find_sim(env)
        self._markers = _find_compiled_marker_geoms(self.sim)
        if not self._markers:
            raise RuntimeError("No compiled MuJoCo marker geoms were found.")

    def render(
        self,
        env: Any,
        *,
        policy_points: np.ndarray,
        retrievals: list[RetrievalTrajectory],
        generated: GeneratedTrajectory | None,
        fallback_frame: np.ndarray,
    ) -> np.ndarray:
        try:
            self.sim = _find_sim(env)
            self._markers = _find_compiled_marker_geoms(self.sim)
            self._update_markers(policy_points=policy_points, retrievals=retrievals, generated=generated)
            try:
                frames = [
                    self.sim.render(width=self.width, height=self.height, camera_name=camera_name)
                    for camera_name in self.camera_names
                ]
            finally:
                self._hide_markers()
            return np.concatenate([np.rot90(np.asarray(frame, dtype=np.uint8), 2) for frame in frames], axis=1)
        except Exception as exc:
            logger.warning("MuJoCo model-marker trajectory render failed; using OpenCV fallback frame: %s", exc)
            return _stack_fallback_frames(fallback_frame, count=len(self.camera_names))

    def close(self) -> None:
        return None

    def _update_markers(
        self,
        *,
        policy_points: np.ndarray,
        retrievals: list[RetrievalTrajectory],
        generated: GeneratedTrajectory | None,
    ) -> None:
        specs = _build_marker_specs(policy_points=policy_points, retrievals=retrievals, generated=generated)
        model = self.sim.model
        data = self.sim.data
        for marker_idx, marker in enumerate(self._markers):
            if marker_idx < len(specs):
                point, color, radius, alpha = specs[marker_idx]
                data.mocap_pos[marker.mocap_id][:] = point
                model.geom_rgba[marker.geom_id][:] = _rgba(color, alpha)
                model.geom_size[marker.geom_id][0] = radius
            else:
                data.mocap_pos[marker.mocap_id][:] = _MARKER_HIDDEN_POS
                model.geom_rgba[marker.geom_id][:] = np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.sim.forward()

    def _hide_markers(self) -> None:
        model = self.sim.model
        data = self.sim.data
        for marker in self._markers:
            data.mocap_pos[marker.mocap_id][:] = _MARKER_HIDDEN_POS
            model.geom_rgba[marker.geom_id][:] = np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.sim.forward()


@dataclasses.dataclass
class _CompiledMarker:
    geom_id: int
    body_id: int
    mocap_id: int


class MuJoCoTrajectorySceneRenderer:
    def __init__(self, env: Any, *, width: int, height: int, camera_name: str):
        import mujoco

        self._mujoco = mujoco
        self.width = width
        self.height = height
        self.camera_name = camera_name
        self.model, self.data = _find_mujoco_model_data(env)
        self.renderer = mujoco.Renderer(self.model, height=height, width=width)

    def render(
        self,
        env: Any,
        *,
        policy_points: np.ndarray,
        retrievals: list[RetrievalTrajectory],
        generated: GeneratedTrajectory | None,
        fallback_frame: np.ndarray,
    ) -> np.ndarray:
        try:
            _model, self.data = _find_mujoco_model_data(env)
            self.renderer.update_scene(self.data, camera=self.camera_name)
            self._add_policy_geoms(policy_points)
            self._add_retrieval_geoms(retrievals)
            self._add_generated_geoms(generated)
            frame = self.renderer.render()
            return np.asarray(frame, dtype=np.uint8)
        except Exception as exc:
            logger.warning("MuJoCo trajectory render failed; using OpenCV fallback frame: %s", exc)
            return np.asarray(fallback_frame)

    def close(self) -> None:
        close = getattr(self.renderer, "close", None)
        if callable(close):
            close()

    def _add_policy_geoms(self, policy_points: np.ndarray) -> None:
        _add_policy_geoms_to_scene(self._mujoco, self.renderer.scene, policy_points)

    def _add_retrieval_geoms(self, retrievals: list[RetrievalTrajectory]) -> None:
        _add_retrieval_geoms_to_scene(self._mujoco, self.renderer.scene, retrievals)

    def _add_generated_geoms(self, generated: GeneratedTrajectory | None) -> None:
        _add_generated_geoms_to_scene(self._mujoco, self.renderer.scene, generated)


class RobosuiteOffscreenTrajectorySceneRenderer:
    def __init__(self, env: Any, *, width: int, height: int, camera_name: str):
        import mujoco

        self._mujoco = mujoco
        self.width = width
        self.height = height
        self.camera_name = camera_name
        self.sim, self.context = _find_robosuite_offscreen_context(env, required=True)
        self._render_lock = _find_robosuite_render_lock()

    def render(
        self,
        env: Any,
        *,
        policy_points: np.ndarray,
        retrievals: list[RetrievalTrajectory],
        generated: GeneratedTrajectory | None,
        fallback_frame: np.ndarray,
    ) -> np.ndarray:
        try:
            self.sim, self.context = _find_robosuite_offscreen_context(env, required=True)
            with self._render_lock:
                frame = self._render_with_context(policy_points=policy_points, retrievals=retrievals, generated=generated)
            return np.asarray(frame, dtype=np.uint8)
        except Exception as exc:
            logger.warning("Robosuite MuJoCo trajectory render failed; using OpenCV fallback frame: %s", exc)
            return np.asarray(fallback_frame)

    def close(self) -> None:
        return None

    def _render_with_context(
        self,
        *,
        policy_points: np.ndarray,
        retrievals: list[RetrievalTrajectory],
        generated: GeneratedTrajectory | None,
    ) -> np.ndarray:
        mujoco = self._mujoco
        context = self.context
        viewport = mujoco.MjrRect(0, 0, self.width, self.height)

        if self.width > context.con.offWidth or self.height > context.con.offHeight:
            new_width = max(self.width, context.model.vis.global_.offwidth)
            new_height = max(self.height, context.model.vis.global_.offheight)
            context.update_offscreen_size(new_width, new_height)

        camera_id = self.sim.model.camera_name2id(self.camera_name)
        context.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        context.cam.fixedcamid = camera_id

        mujoco.mjv_updateScene(
            _unwrap_mujoco_object(context.model, ("_model", "ptr")),
            _unwrap_mujoco_object(context.data, ("_data", "ptr")),
            context.vopt,
            context.pert,
            context.cam,
            mujoco.mjtCatBit.mjCAT_ALL,
            context.scn,
        )
        _add_policy_geoms_to_scene(mujoco, context.scn, policy_points)
        _add_retrieval_geoms_to_scene(mujoco, context.scn, retrievals)
        _add_generated_geoms_to_scene(mujoco, context.scn, generated)
        mujoco.mjr_render(viewport=viewport, scn=context.scn, con=context.con)
        return context.read_pixels(self.width, self.height)


def _add_policy_geoms_to_scene(mujoco: Any, scene: Any, policy_points: np.ndarray) -> None:
    if len(policy_points) < 2:
        return
    _add_polyline_geoms(
        mujoco,
        scene,
        policy_points,
        rgba=_rgba(_POLICY_COLOR, 0.85),
        width=0.006,
        max_segments=120,
    )
    _add_sphere_geom(
        mujoco,
        scene,
        policy_points[-1],
        rgba=_rgba(_POLICY_HEAD_COLOR, 0.95),
        radius=0.018,
    )


def _add_retrieval_geoms_to_scene(mujoco: Any, scene: Any, retrievals: list[RetrievalTrajectory]) -> None:
    for retrieval in retrievals:
        points = np.asarray(retrieval.points, dtype=np.float32)
        if len(points) == 0:
            continue
        color = _color_for_rank(retrieval.rank)
        alpha = 0.86 if retrieval.rank == 0 else 0.56
        _add_polyline_geoms(
            mujoco,
            scene,
            points,
            rgba=_rgba(color, alpha),
            width=0.0045 if retrieval.rank == 0 else 0.0035,
            max_segments=80,
        )
        _add_sphere_geom(
            mujoco,
            scene,
            points[0],
            rgba=_rgba(color, 0.95),
            radius=0.014,
        )


def _add_generated_geoms_to_scene(mujoco: Any, scene: Any, generated: GeneratedTrajectory | None) -> None:
    if generated is None:
        return
    points = np.asarray(generated.points, dtype=np.float32)
    if len(points) < 2:
        return
    _add_polyline_geoms(
        mujoco,
        scene,
        points,
        rgba=_rgba(_GENERATED_COLOR, 0.9),
        width=0.0065,
        max_segments=80,
    )
    _add_sphere_geom(
        mujoco,
        scene,
        points[-1],
        rgba=_rgba(_GENERATED_HEAD_COLOR, 0.98),
        radius=0.017,
    )


def _build_marker_specs(
    *,
    policy_points: np.ndarray,
    retrievals: list[RetrievalTrajectory],
    generated: GeneratedTrajectory | None,
) -> list[tuple[np.ndarray, tuple[int, int, int], float, float]]:
    specs: list[tuple[np.ndarray, tuple[int, int, int], float, float]] = []
    policy_points = _downsample_points(np.asarray(policy_points, dtype=np.float32), max_points=70)
    for idx, point in enumerate(policy_points):
        color = _POLICY_HEAD_COLOR if idx == len(policy_points) - 1 else _POLICY_COLOR
        radius = 0.017 if idx == len(policy_points) - 1 else 0.009
        specs.append((point, color, radius, 0.92))

    if generated is not None:
        generated_points = _downsample_points(np.asarray(generated.points, dtype=np.float32), max_points=24)
        for idx, point in enumerate(generated_points):
            color = _GENERATED_HEAD_COLOR if idx == len(generated_points) - 1 else _GENERATED_COLOR
            radius = 0.017 if idx == len(generated_points) - 1 else 0.009
            specs.append((point, color, radius, 0.93))

    for retrieval in retrievals:
        points = _downsample_points(np.asarray(retrieval.points, dtype=np.float32), max_points=20)
        color = _color_for_rank(retrieval.rank)
        alpha = 0.88 if retrieval.rank == 0 else 0.58
        for idx, point in enumerate(points):
            radius = 0.014 if idx == 0 else 0.008
            specs.append((point, color, radius, alpha))
    return specs


def _downsample_points(points: np.ndarray, *, max_points: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if len(points) <= max_points:
        return points
    indices = np.unique(np.linspace(0, len(points) - 1, max_points).round().astype(np.int32))
    return points[indices]


def _parse_camera_names(camera_name: str) -> list[str]:
    names = [name.strip() for name in str(camera_name).split(",") if name.strip()]
    return names or ["agentview"]


def _add_trajectory_cameras(worldbody: ET.Element) -> None:
    for existing in list(worldbody.findall("camera")):
        name = existing.get("name", "")
        if name.startswith(f"{_CAMERA_PREFIX}_"):
            worldbody.remove(existing)

    camera_specs = {
        f"{_CAMERA_PREFIX}_left": (
            np.asarray([0.18, 0.92, 1.24], dtype=np.float32),
            np.asarray([0.08, 0.00, 0.88], dtype=np.float32),
        ),
        f"{_CAMERA_PREFIX}_front": (
            np.asarray([0.86, 0.00, 1.28], dtype=np.float32),
            np.asarray([0.02, 0.00, 0.88], dtype=np.float32),
        ),
        f"{_CAMERA_PREFIX}_right": (
            np.asarray([0.18, -0.92, 1.24], dtype=np.float32),
            np.asarray([0.08, 0.00, 0.88], dtype=np.float32),
        ),
    }
    for name, (pos, target) in camera_specs.items():
        ET.SubElement(
            worldbody,
            "camera",
            {
                "name": name,
                "pos": _array_to_mjcf(pos),
                "xyaxes": _array_to_mjcf(_camera_xyaxes(pos, target)),
                "fovy": "52",
            },
        )


def _camera_xyaxes(pos: np.ndarray, target: np.ndarray) -> np.ndarray:
    forward = np.asarray(target - pos, dtype=np.float64)
    forward = forward / np.linalg.norm(forward)
    world_up = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-6:
        right = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)
    return np.concatenate([right, up]).astype(np.float32)


def _stack_fallback_frames(frame: np.ndarray, *, count: int) -> np.ndarray:
    frame = np.asarray(frame, dtype=np.uint8)
    if count <= 1:
        return frame
    return np.concatenate([frame.copy() for _ in range(count)], axis=1)


def _find_xml_processor_target(env: Any) -> Any | None:
    for root in _env_candidates(env):
        if callable(getattr(root, "set_xml_processor", None)):
            return root
    return None


def _find_sim(env: Any) -> Any:
    for root in _env_candidates(env):
        sim = getattr(root, "sim", None)
        if sim is not None:
            return sim
    raise RuntimeError("Could not find MuJoCo sim on the LIBERO environment.")


def _find_compiled_marker_geoms(sim: Any) -> list[_CompiledMarker]:
    markers = []
    marker_idx = 0
    while True:
        geom_name = f"{_MARKER_PREFIX}_geom_{marker_idx:04d}"
        body_name = f"{_MARKER_PREFIX}_body_{marker_idx:04d}"
        try:
            geom_id = int(sim.model.geom_name2id(geom_name))
            body_id = int(sim.model.body_name2id(body_name))
        except Exception:
            break
        mocap_id = int(sim.model.body_mocapid[body_id])
        if mocap_id >= 0:
            markers.append(_CompiledMarker(geom_id=geom_id, body_id=body_id, mocap_id=mocap_id))
        marker_idx += 1
    return markers


def _count_compiled_marker_geoms(env: Any) -> int:
    try:
        return len(_find_compiled_marker_geoms(_find_sim(env)))
    except Exception:
        return 0


def _array_to_mjcf(values: np.ndarray) -> str:
    return " ".join(f"{float(value):.6g}" for value in values)


def _find_mujoco_model_data(env: Any) -> tuple[Any, Any]:
    for root in _env_candidates(env):
        sim = getattr(root, "sim", None)
        if sim is not None:
            model = getattr(sim, "model", None)
            data = getattr(sim, "data", None)
            if model is not None and data is not None:
                return _unwrap_mujoco_object(model, ("_model", "ptr")), _unwrap_mujoco_object(data, ("_data", "ptr"))

        model = getattr(root, "model", None)
        data = getattr(root, "data", None)
        if model is not None and data is not None:
            return _unwrap_mujoco_object(model, ("_model", "ptr")), _unwrap_mujoco_object(data, ("_data", "ptr"))

    raise RuntimeError("Could not find MuJoCo model/data on the LIBERO environment.")


def _find_robosuite_offscreen_context(env: Any, *, required: bool = False) -> tuple[Any, Any] | None:
    for root in _env_candidates(env):
        sim = getattr(root, "sim", None)
        context = getattr(sim, "_render_context_offscreen", None) if sim is not None else None
        if context is not None:
            return sim, context
    if required:
        raise RuntimeError("Could not find robosuite offscreen render context.")
    return None


def _find_robosuite_render_lock() -> threading.Lock:
    try:
        from robosuite.utils import binding_utils

        lock = getattr(binding_utils, "_MjSim_render_lock", None)
        if lock is not None:
            return lock
    except Exception:
        pass
    return threading.Lock()


def _unwrap_mujoco_object(obj: Any, attr_names: tuple[str, ...]) -> Any:
    seen = set()
    current = obj
    while id(current) not in seen:
        seen.add(id(current))
        for attr_name in attr_names:
            child = getattr(current, attr_name, None)
            if child is not None and child is not current:
                current = child
                break
        else:
            return current
    return current


def _env_candidates(env: Any) -> list[Any]:
    candidates = [env]
    for attr in ("env", "_env", "base_env", "wrapped_env"):
        child = getattr(env, attr, None)
        if child is not None:
            candidates.append(child)
    return candidates


def _add_polyline_geoms(
    mujoco: Any,
    scene: Any,
    points: np.ndarray,
    *,
    rgba: np.ndarray,
    width: float,
    max_segments: int,
) -> None:
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 2:
        return
    segments = trajectory_segments(points, max_segments=max_segments)
    for start, end in segments:
        if scene.ngeom >= scene.maxgeom:
            logger.warning("MuJoCo scene maxgeom reached; remaining trajectory geoms will be omitted.")
            return
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, width, start, end)
        geom.rgba[:] = rgba
        scene.ngeom += 1


def _add_sphere_geom(mujoco: Any, scene: Any, point: np.ndarray, *, rgba: np.ndarray, radius: float) -> None:
    if scene.ngeom >= scene.maxgeom:
        logger.warning("MuJoCo scene maxgeom reached; trajectory sphere will be omitted.")
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.asarray([radius, radius, radius], dtype=np.float64),
        np.asarray(point, dtype=np.float64),
        np.eye(3, dtype=np.float64).reshape(-1),
        rgba,
    )
    scene.ngeom += 1


def trajectory_segments(points: np.ndarray, *, max_segments: int) -> list[tuple[np.ndarray, np.ndarray]]:
    if len(points) < 2:
        return []
    if len(points) - 1 <= max_segments:
        selected = points
    else:
        indices = np.unique(np.linspace(0, len(points) - 1, max_segments + 1).round().astype(np.int32))
        selected = points[indices]
    return [(selected[i], selected[i + 1]) for i in range(len(selected) - 1)]


def _rgba(color: tuple[int, int, int], alpha: float) -> np.ndarray:
    return np.asarray([color[0] / 255.0, color[1] / 255.0, color[2] / 255.0, alpha], dtype=np.float32)


def _color_for_rank(rank: int) -> tuple[int, int, int]:
    if rank < len(_RETRIEVAL_COLORS):
        return _RETRIEVAL_COLORS[rank]
    return _RETRIEVAL_COLORS[-1]


def _project_points(
    points: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray],
    frame_shape: tuple[int, ...],
    margin_px: int,
) -> np.ndarray:
    low, high = bounds
    height, width = frame_shape[:2]
    xy = np.asarray(points, dtype=np.float32)[:, :2]
    span = np.maximum(high[:2] - low[:2], 1e-6)
    normalized = (xy - low[:2]) / span
    x = margin_px + normalized[:, 0] * max(width - 2 * margin_px, 1)
    y = height - (margin_px + normalized[:, 1] * max(height - 2 * margin_px, 1))
    return np.stack([x, y], axis=-1).astype(np.int32)


def _draw_polyline(frame: np.ndarray, points: np.ndarray, color: tuple[int, int, int], *, thickness: int, alpha: float, cv2: Any) -> None:
    if len(points) < 2:
        return
    overlay = frame.copy()
    cv2.polylines(overlay, [points.reshape(-1, 1, 2)], isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0.0, dst=frame)


def _draw_circle(frame: np.ndarray, point: np.ndarray, *, radius: int, color: tuple[int, int, int], alpha: float, cv2: Any) -> None:
    overlay = frame.copy()
    cv2.circle(overlay, tuple(int(v) for v in point), radius, color, thickness=-1, lineType=cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0.0, dst=frame)


def _draw_legend(frame: np.ndarray, retrievals: list[RetrievalTrajectory], *, cv2: Any) -> None:
    overlay = frame.copy()
    cv2.rectangle(overlay, (7, 7), (188, 43 + 17 * min(len(retrievals), 5)), (0, 0, 0), thickness=-1)
    cv2.addWeighted(overlay, 0.38, frame, 0.62, 0.0, dst=frame)
    cv2.putText(frame, "policy/executed", (14, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, _POLICY_COLOR, 1, cv2.LINE_AA)
    cv2.putText(frame, "model generated", (14, 39), cv2.FONT_HERSHEY_SIMPLEX, 0.42, _GENERATED_COLOR, 1, cv2.LINE_AA)
    y = 56
    for retrieval in retrievals[:5]:
        color = _color_for_rank(retrieval.rank)
        label = f"retrieved r{retrieval.rank}"
        if retrieval.distance is not None:
            label += f" d={retrieval.distance:.1f}"
        cv2.putText(frame, label, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
        y += 17


def _find_marker_hook(env: Any) -> str | None:
    candidates = (
        ("viewer", "add_marker"),
        ("sim.viewer", "add_marker"),
        ("env.viewer", "add_marker"),
    )
    for path, method in candidates:
        obj = env
        for attr in path.split("."):
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None and callable(getattr(obj, method, None)):
            return f"{path}.{method}"
    return None
