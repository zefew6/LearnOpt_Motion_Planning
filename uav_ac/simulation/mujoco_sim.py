from collections.abc import Callable
from pathlib import Path
import time

import mujoco
import numpy as np

from uav_ac.planning.corridor.firi import FIRI3D, FIRIRegion
from uav_ac.quadrotor.quad import Quad
from uav_ac.visualization import CorridorMeshVisualizer, add_corridor_mesh_pool

from .recording import Mp4Recorder, default_camera


ENU_TO_NED = np.diag([1.0, -1.0, -1.0])
DEFAULT_SCENE_PATH = Path(__file__).parent / "models" / "lab_course.xml"
GCS_BUILDING_SCENE_PATH = Path(__file__).parent / "models" / "gcs_building.xml"
OPEN_FIELD_SCENE_PATH = Path(__file__).parent / "models" / "open_field.xml"
TRAJECTORY_SEGMENT_COUNT = 200
TRAJECTORY_COLOR = np.array([1.0, 0.25, 0.05, 0.35])
ACTUAL_TRAJECTORY_SEGMENT_COUNT = 200
ACTUAL_TRAJECTORY_COLOR = np.array([0.1, 0.4, 1.0, 0.9])
ACTUAL_TRAJECTORY_SAMPLE_INTERVAL = 0.05
TAKEOFF_HEIGHT = 0.1


def mujoco_to_ned_state(
        position: np.ndarray,
        quaternion: np.ndarray,
        velocity: np.ndarray,
) -> np.ndarray:
    """
    Convert MuJoCo ENU/FLU free-joint state to the NED/FRD convention.

    :param position: world position in ENU
    :param quaternion: FLU-to-ENU quaternion in scalar-first order
    :param velocity: world linear velocity followed by FLU angular velocity
    :return: controller state vector
    """
    position = _vector(position, 3, "position")
    quaternion = _vector(quaternion, 4, "quaternion")
    velocity = _vector(velocity, 6, "velocity")
    quaternion_norm = np.linalg.norm(quaternion)
    if quaternion_norm == 0:
        raise ValueError("MuJoCo quaternion cannot be zero")

    state = np.empty(13)
    state[:3] = ENU_TO_NED @ position
    state[3:7] = quaternion / quaternion_norm * np.array([1.0, 1.0, -1.0, -1.0])
    state[7:10] = ENU_TO_NED @ velocity[:3]
    state[10:13] = ENU_TO_NED @ velocity[3:]
    return state


class MujocoSimulation:
    """Simulate the quadrotor rigid-body dynamics and collisions with MuJoCo."""

    def __init__(
            self,
            model_path: str | Path = DEFAULT_SCENE_PATH,
            *,
            record_actual_trajectory: bool = True,
    ):
        """
        Load the vehicle, mission and obstacle geometry from a MuJoCo scene.

        :param model_path: MJCF scene containing the required quadrotor and mission data
        :param record_actual_trajectory: update the blue flown-path geometry while stepping
        """
        specification = mujoco.MjSpec.from_file(str(model_path))
        add_corridor_mesh_pool(specification)
        self.model = specification.compile()
        self.data = mujoco.MjData(self.model)
        self._body_id = _named_id(self.model, mujoco.mjtObj.mjOBJ_BODY, "quadrotor")
        self._body_geom_id = _named_id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "body")
        self._tracking_beacon_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "drone_tracking_beacon")
        self._ground_geom_id = _named_id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
        self._rotor_site_ids = np.array([_named_id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, f"rotor_{index}") for index in range(4)])
        self._rotor_spin_directions = self.model.site_user[self._rotor_site_ids, 0].copy()
        self._trajectory_segment_ids = np.array([_named_id(
            self.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            f"trajectory_segment{index:03d}",
        ) for index in range(TRAJECTORY_SEGMENT_COUNT)])
        self._actual_trajectory_segment_ids = np.array([_named_id(
            self.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            f"actual_trajectory_segment{index:03d}",
        ) for index in range(ACTUAL_TRAJECTORY_SEGMENT_COUNT)])
        self._corridor_visualizer = CorridorMeshVisualizer(self.model)
        # Compatibility aliases for existing diagnostics and downstream code.
        self._corridor_region_ids = self._corridor_visualizer.region_ids
        self._corridor_mesh_ids = self._corridor_visualizer.mesh_ids
        self.quad = _create_quad(self.model, self._body_id, self._rotor_site_ids)
        self._collision_detected = False
        self._has_taken_off = False
        self._external_force_world = np.zeros(3)
        self._actual_trajectory_positions = []
        self._next_actual_trajectory_sample_time = 0.0
        self._record_actual_trajectory_enabled = bool(record_actual_trajectory)

        mujoco.mj_forward(self.model, self.data)
        self._sync_quad_state()
        self.start_position = self.quad.position.copy()
        goal_id = _named_id(self.model, mujoco.mjtObj.mjOBJ_SITE, "goal")
        self.goal_position = ENU_TO_NED @ self.data.site_xpos[goal_id]
        self.mission_waypoints = _extract_mission_waypoints(
            self.model, self.data, self.start_position, self.goal_position)
        self.gcs_guide_paths = _extract_gcs_guide_paths(self.model, self.data)
        self.space_limits = _numeric(self.model, "planning_bounds", 6).reshape(2, 3)
        self.obstacles = _extract_obstacles(self.model, self.data)
        if self.has_collision:
            raise ValueError("quadrotor starts in collision")

    @property
    def has_collision(self) -> bool:
        """Return whether the quadrotor currently touches world geometry."""
        return self.data.ncon > 0

    @property
    def collision_detected(self) -> bool:
        """Return whether any collision has occurred since initialization."""
        return self._collision_detected

    def set_trajectory_visualization(self, positions: np.ndarray) -> None:
        """Display a sampled NED trajectory using non-colliding MuJoCo capsules."""
        positions = np.asarray(positions, dtype=float)
        if positions.ndim != 2 or positions.shape[1] != 3 or len(positions) < 2:
            raise ValueError("trajectory positions must have shape (n, 3) with n >= 2")
        if not np.all(np.isfinite(positions)):
            raise ValueError("trajectory positions must be finite")

        self._set_trajectory_segments(
            positions, self._trajectory_segment_ids, TRAJECTORY_COLOR)
        mujoco.mj_forward(self.model, self.data)

    def set_convex_polyhedra_visualization(self, regions: list[FIRIRegion]) -> int:
        """Draw convex regions as translucent, non-colliding solid meshes."""
        count = self._corridor_visualizer.set_regions(regions, ENU_TO_NED)
        mujoco.mj_forward(self.model, self.data)
        return count

    def set_external_force_world(self, force: np.ndarray) -> None:
        """Set a persistent world-frame disturbance force applied at the vehicle COM."""
        self._external_force_world = _vector(force, 3, "external_force_world").copy()

    def set_goal_position(self, position_ned: np.ndarray) -> None:
        """Move the visible goal site and update the public NED goal position."""
        position_ned = _vector(position_ned, 3, "goal_position")
        goal_id = _named_id(self.model, mujoco.mjtObj.mjOBJ_SITE, "goal")
        self.model.site_pos[goal_id] = ENU_TO_NED @ position_ned
        self.goal_position = position_ned.copy()
        mujoco.mj_forward(self.model, self.data)

    def reset(
            self,
            state_ned: np.ndarray | None = None,
            motor_speeds: np.ndarray | None = None,
    ) -> np.ndarray:
        """Reset all MuJoCo and adapter state, optionally to an NED/FRD snapshot.

        ``state_ned`` uses the same 13-element layout as :attr:`Quad.X`.
        ``motor_speeds`` contains the four physical rotor speeds in rad/s.  The
        returned state is a copy, so callers cannot mutate the simulation by
        retaining it.
        """
        if state_ned is not None:
            state_ned = _vector(state_ned, 13, "state_ned")
            quaternion_norm = np.linalg.norm(state_ned[3:7])
            if quaternion_norm == 0.0:
                raise ValueError("MuJoCo state_ned quaternion cannot be zero")
            state_ned = state_ned.copy()
            state_ned[3:7] /= quaternion_norm

        if motor_speeds is None:
            motor_speeds = np.zeros(4)
        else:
            motor_speeds = _vector(motor_speeds, 4, "motor_speeds")
            maximum_speed = np.sqrt(self.quad.max_thrust / self.quad.kf)
            if np.any(motor_speeds < 0.0) or np.any(motor_speeds > maximum_speed + 1.0e-9):
                raise ValueError("MuJoCo motor_speeds must be within physical rotor limits")
            motor_speeds = np.clip(motor_speeds, 0.0, maximum_speed)

        mujoco.mj_resetData(self.model, self.data)
        if state_ned is not None:
            self.data.qpos[:3] = ENU_TO_NED @ state_ned[:3]
            self.data.qpos[3:7] = state_ned[3:7] * np.array([1.0, 1.0, -1.0, -1.0])
            self.data.qvel[:3] = ENU_TO_NED @ state_ned[7:10]
            self.data.qvel[3:6] = ENU_TO_NED @ state_ned[10:13]
        mujoco.mj_forward(self.model, self.data)
        self._reset_runtime_state(motor_speeds)
        self._record_collisions()
        return self.quad.X.copy()

    def get_planning_obstacle_points(
            self,
            spacing: float = 0.5,
            padding: float = 0.15,
    ) -> np.ndarray:
        """Sample all visible static course obstacles in the NED frame."""
        points_enu = []
        obstacle_prefixes = ("obstacle_", "gate_", "ring_")
        for geom_id in range(self.model.ngeom):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            if name is None or not name.startswith(obstacle_prefixes):
                continue
            geom_type = self.model.geom_type[geom_id]
            center = self.data.geom_xpos[geom_id]
            rotation = self.data.geom_xmat[geom_id].reshape(3, 3)
            if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
                half_size = self.model.geom_size[geom_id] + padding
                local_bounds = np.array([
                    -half_size[0], half_size[0],
                    -half_size[1], half_size[1],
                    -half_size[2], half_size[2],
                ])
                local_points = FIRI3D.sample_aabb_surfaces(
                    local_bounds[None, :], spacing=spacing)
            elif geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
                radius = self.model.geom_size[geom_id, 0] + padding
                half_length = self.model.geom_size[geom_id, 1]
                angle_count = max(8, int(np.ceil(2.0 * np.pi * radius / spacing)))
                axial_count = max(2, int(np.ceil(2.0 * half_length / spacing)) + 1)
                angles = np.linspace(0.0, 2.0 * np.pi, angle_count, endpoint=False)
                axial_positions = np.linspace(-half_length, half_length, axial_count)
                local_points = np.array([
                    [radius * np.cos(angle), radius * np.sin(angle), axial_position]
                    for axial_position in axial_positions
                    for angle in angles
                ])
                local_points = np.vstack((
                    local_points,
                    [0.0, 0.0, -half_length - radius],
                    [0.0, 0.0, half_length + radius],
                ))
            else:
                continue
            points_enu.append(local_points @ rotation.T + center)

        if not points_enu:
            return np.zeros((0, 3))
        points_ned = np.vstack(points_enu) @ ENU_TO_NED
        return np.unique(np.round(points_ned, decimals=10), axis=0)

    def sample_free_space(
            self,
            spacing: np.ndarray | tuple[float, float, float] = (1.5, 1.5, 1.0),
            clearance: float = 0.15,
    ) -> np.ndarray:
        """Return collision-checked NED grid samples in the planning volume."""
        spacing = np.asarray(spacing, dtype=float)
        if spacing.shape != (3,) or np.any(spacing <= 0.0) or clearance < 0.0:
            raise ValueError("spacing must be positive in 3D and clearance non-negative")
        axes = [
            np.arange(low + step / 2.0, high, step)
            for low, high, step in zip(self.space_limits[0], self.space_limits[1], spacing)
        ]
        grid = np.meshgrid(*axes, indexing="ij")
        points_ned = np.column_stack([axis.ravel() for axis in grid])
        points_enu = points_ned @ ENU_TO_NED
        collision_free = np.ones(len(points_ned), dtype=bool)

        obstacle_prefixes = ("obstacle_", "gate_", "ring_")
        for geom_id in range(self.model.ngeom):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            if name is None or not name.startswith(obstacle_prefixes):
                continue
            center = self.data.geom_xpos[geom_id]
            rotation = self.data.geom_xmat[geom_id].reshape(3, 3)
            local_points = (points_enu - center) @ rotation
            geom_type = self.model.geom_type[geom_id]
            if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
                half_size = self.model.geom_size[geom_id] + clearance
                colliding = np.all(np.abs(local_points) <= half_size, axis=1)
            elif geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
                radius = self.model.geom_size[geom_id, 0] + clearance
                half_length = self.model.geom_size[geom_id, 1]
                closest_z = np.clip(local_points[:, 2], -half_length, half_length)
                offsets = local_points.copy()
                offsets[:, 2] -= closest_z
                colliding = np.linalg.norm(offsets, axis=1) <= radius
            else:
                continue
            collision_free &= ~colliding
        return points_ned[collision_free]

    def _set_trajectory_segments(
            self,
            positions: np.ndarray,
            segment_ids: np.ndarray,
            color: np.ndarray,
    ) -> None:
        max_points = len(segment_ids) + 1
        if len(positions) > max_points:
            sample_indices = np.linspace(0, len(positions) - 1, max_points).round().astype(int)
            positions = positions[sample_indices]

        positions_enu = positions @ ENU_TO_NED
        self.model.geom_rgba[segment_ids, 3] = 0.0
        segment_index = 0
        for start, end in zip(positions_enu[:-1], positions_enu[1:]):
            direction = end - start
            length = np.linalg.norm(direction)
            if length == 0:
                continue

            geom_id = segment_ids[segment_index]
            self.model.geom_pos[geom_id] = (start + end) / 2
            self.model.geom_size[geom_id, 1] = length / 2
            self.model.geom_rbound[geom_id] = self.model.geom_size[geom_id, 0] + length / 2
            mujoco.mju_quatZ2Vec(self.model.geom_quat[geom_id], direction)
            self.model.geom_sameframe[geom_id] = mujoco.mjtSameFrame.mjSAMEFRAME_NONE
            self.model.geom_rgba[geom_id] = color
            segment_index += 1

    def step(self) -> np.ndarray:
        """Advance MuJoCo by one inner-loop time step using current rotor speeds."""
        self._apply_rotor_forces()
        if self._tracking_beacon_id >= 0:
            self.model.geom_pos[self._tracking_beacon_id, :2] = (
                self.data.xpos[self._body_id, :2])
        mujoco.mj_step(self.model, self.data)
        self._sync_quad_state()
        self._record_collisions()
        self._record_actual_trajectory()
        return self.quad.X.copy()

    def run_interactive(
            self,
            control_step: Callable[[], None],
            reset_control: Callable[[], None] | None = None,
    ) -> None:
        """
        Run the native MuJoCo viewer while the supplied controller drives the rotors.

        :param control_step: one inner-loop update of the existing flight controller
        :param reset_control: reset the controller when the viewer resets the simulation
        """
        from mujoco import viewer

        last_simulation_time = None
        with viewer.launch_passive(self.model, self.data) as viewer_handle:
            # The managed viewer owns its camera initialization and may ignore
            # the MJCF defaults.  Initialize the exposed passive-viewer camera
            # explicitly from <statistic> and <visual><global> in the model.
            with viewer_handle.lock():
                mujoco.mjv_defaultFreeCamera(self.model, viewer_handle.cam)
            viewer_handle.sync()

            while viewer_handle.is_running():
                step_start = time.perf_counter()
                if (last_simulation_time is not None
                        and self.data.time < last_simulation_time):
                    self._reset_runtime_state()
                    if reset_control is not None:
                        reset_control()

                control_step()
                self.step()
                last_simulation_time = self.data.time
                viewer_handle.sync()

                remaining_step_time = self.model.opt.timestep - (
                    time.perf_counter() - step_start)
                if remaining_step_time > 0:
                    time.sleep(remaining_step_time)

    def run_recorded(
            self,
            control_step: Callable[[], None],
            max_steps: int,
            output_path: str | Path,
            *,
            fps: float = 30.0,
            width: int = 1280,
            height: int = 720,
            camera: mujoco.MjvCamera | None = None,
    ) -> int:
        """Run a fixed-duration experiment and save an offscreen MP4 recording.

        Frames are sampled from simulation time, while control and physics still
        execute at the model timestep.  This keeps the experiment deterministic
        and avoids opening the native MuJoCo viewer.

        :return: number of video frames written
        """
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        if fps <= 0.0 or not np.isfinite(fps):
            raise ValueError("fps must be positive and finite")
        frame_period = 1.0 / fps
        old_offwidth = int(self.model.vis.global_.offwidth)
        old_offheight = int(self.model.vis.global_.offheight)
        self.model.vis.global_.offwidth = max(old_offwidth, width)
        self.model.vis.global_.offheight = max(old_offheight, height)
        renderer = None
        camera = default_camera(self.model) if camera is None else camera

        def render_frame() -> np.ndarray:
            renderer.update_scene(self.data, camera=camera)
            return renderer.render()

        try:
            renderer = mujoco.Renderer(self.model, height=height, width=width)
            with Mp4Recorder(output_path, width, height, fps) as recorder:
                recorder.write(render_frame())
                next_frame_time = frame_period
                for _ in range(max_steps):
                    control_step()
                    self.step()
                    if self.data.time + 1.0e-12 >= next_frame_time:
                        recorder.write(render_frame())
                        next_frame_time += frame_period
        finally:
            if renderer is not None:
                renderer.close()
            self.model.vis.global_.offwidth = old_offwidth
            self.model.vis.global_.offheight = old_offheight
        return recorder.frame_count

    def _reset_runtime_state(self, motor_speeds: np.ndarray | None = None) -> None:
        motor_speeds = np.zeros(4) if motor_speeds is None else motor_speeds
        self.quad.omega[:] = motor_speeds
        self.quad.omega_command[:] = motor_speeds
        self.data.qfrc_applied.fill(0.0)
        self._external_force_world.fill(0.0)
        self._collision_detected = False
        self._has_taken_off = self.data.xpos[self._body_id, 2] >= TAKEOFF_HEIGHT
        self._actual_trajectory_positions.clear()
        self._next_actual_trajectory_sample_time = 0.0
        self.model.geom_rgba[self._actual_trajectory_segment_ids, 3] = 0.0
        self._sync_quad_state()

    def _record_actual_trajectory(self) -> None:
        if not self._record_actual_trajectory_enabled:
            return
        if self.data.time < self._next_actual_trajectory_sample_time:
            return

        self._actual_trajectory_positions.append(self.quad.position.copy())
        self._next_actual_trajectory_sample_time = (
            self.data.time + ACTUAL_TRAJECTORY_SAMPLE_INTERVAL)
        if len(self._actual_trajectory_positions) < 2:
            return

        self._set_trajectory_segments(
            np.asarray(self._actual_trajectory_positions),
            self._actual_trajectory_segment_ids,
            ACTUAL_TRAJECTORY_COLOR,
        )

    def _record_collisions(self) -> None:
        if self.data.xpos[self._body_id, 2] >= TAKEOFF_HEIGHT:
            self._has_taken_off = True

        takeoff_contact = {self._ground_geom_id, self._body_geom_id}
        for contact in self.data.contact:
            contact_geometries = {contact.geom1, contact.geom2}
            if not self._has_taken_off and contact_geometries == takeoff_contact:
                continue
            self._collision_detected = True
            return

    def _apply_rotor_forces(self) -> None:
        self.data.qfrc_applied[:] = 0.0
        body_rotation = self.data.xmat[self._body_id].reshape(3, 3)
        rotor_forces = self.quad.kf * self.quad.omega ** 2

        for index, force in enumerate(rotor_forces):
            force_world = body_rotation @ np.array([0.0, 0.0, force])
            torque_body = np.array([
                0.0, 0.0, self._rotor_spin_directions[index] * self.quad.kappa * force
            ])
            torque_world = body_rotation @ torque_body
            mujoco.mj_applyFT(
                self.model,
                self.data,
                force_world,
                torque_world,
                self.data.site_xpos[self._rotor_site_ids[index]],
                self._body_id,
                self.data.qfrc_applied,
            )
        if np.any(self._external_force_world):
            mujoco.mj_applyFT(
                self.model,
                self.data,
                self._external_force_world,
                np.zeros(3),
                self.data.xpos[self._body_id],
                self._body_id,
                self.data.qfrc_applied,
            )

    def _sync_quad_state(self) -> None:
        self.quad.X = mujoco_to_ned_state(
            self.data.qpos[:3], self.data.qpos[3:7], self.data.qvel[:6])


def _create_quad(model: mujoco.MjModel, body_id: int, rotor_site_ids: np.ndarray) -> Quad:
    rotor_positions = model.site_pos[rotor_site_ids]
    arm_lengths = np.abs(rotor_positions[:, :2])
    if not np.allclose(arm_lengths, arm_lengths[0, 0]):
        raise ValueError("MuJoCo rotor sites must use a symmetric X configuration")

    gravity = np.linalg.norm(model.opt.gravity)
    if gravity == 0:
        raise ValueError("MuJoCo gravity must be non-zero")

    return Quad(
        g=gravity,
        dt=model.opt.timestep,
        mass=model.body_mass[body_id],
        inertia=model.body_inertia[body_id],
        arm_length=arm_lengths[0, 0],
        force_coefficient=_numeric(model, "rotor_force_coefficient", 1)[0],
        drag_to_thrust=_numeric(model, "rotor_drag_to_thrust", 1)[0],
        thrust_limits=_numeric(model, "rotor_thrust_limits", 2),
        motor_time_constants=_numeric(model, "motor_time_constants", 2),
        flight_limits=_numeric(model, "flight_limits", 5),
    )


def _extract_obstacles(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    obstacles = []
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        if name is None or not name.startswith("obstacle_"):
            continue
        if model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_BOX:
            raise ValueError(f"MuJoCo planning obstacle '{name}' must be an axis-aligned box")
        if not np.allclose(data.geom_xmat[geom_id].reshape(3, 3), np.eye(3)):
            raise ValueError(f"MuJoCo planning obstacle '{name}' must be axis-aligned")

        center_ned = ENU_TO_NED @ data.geom_xpos[geom_id]
        half_size = model.geom_size[geom_id]
        obstacles.append(np.array([
            center_ned[0] - half_size[0], center_ned[0] + half_size[0],
            center_ned[1] - half_size[1], center_ned[1] + half_size[1],
            center_ned[2] - half_size[2], center_ned[2] + half_size[2],
        ]))
    return np.asarray(obstacles, dtype=float).reshape(-1, 6)


def _extract_mission_waypoints(
        model: mujoco.MjModel,
        data: mujoco.MjData,
        start_position: np.ndarray,
        goal_position: np.ndarray,
) -> np.ndarray:
    waypoint_ids = []
    for site_id in range(model.nsite):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, site_id)
        if name is not None and name.startswith("waypoint_"):
            waypoint_ids.append((name, site_id))

    waypoint_ids.sort()
    expected_names = [f"waypoint_{index:02d}" for index in range(len(waypoint_ids))]
    if [name for name, _ in waypoint_ids] != expected_names:
        raise ValueError("MuJoCo mission waypoints must be consecutively numbered from waypoint_00")
    if not waypoint_ids:
        raise ValueError("MuJoCo scene must define at least one mandatory waypoint")

    mandatory_waypoints = np.array([
        ENU_TO_NED @ data.site_xpos[site_id] for _, site_id in waypoint_ids
    ])
    return np.vstack((start_position, mandatory_waypoints, goal_position))


def _extract_gcs_guide_paths(
        model: mujoco.MjModel,
        data: mujoco.MjData,
) -> list[np.ndarray]:
    """Read consecutively numbered GCS corridor guide paths from scene sites."""
    grouped: dict[int, list[tuple[int, int]]] = {}
    prefix = "gcs_route_"
    for site_id in range(model.nsite):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, site_id)
        if name is None or not name.startswith(prefix):
            continue
        parts = name[len(prefix):].split("_")
        if len(parts) != 2 or not all(part.isdigit() for part in parts):
            raise ValueError(
                "GCS guide sites must use names gcs_route_<route>_<point>")
        route_index, point_index = map(int, parts)
        grouped.setdefault(route_index, []).append((point_index, site_id))
    if not grouped:
        return []
    if sorted(grouped) != list(range(len(grouped))):
        raise ValueError("GCS guide routes must be consecutively numbered from zero")
    routes = []
    for route_index in range(len(grouped)):
        points = sorted(grouped[route_index])
        if [index for index, _ in points] != list(range(len(points))):
            raise ValueError(
                f"GCS guide route {route_index} points must be consecutively numbered")
        if len(points) < 2:
            raise ValueError("each GCS guide route must contain at least two points")
        routes.append(np.array([
            ENU_TO_NED @ data.site_xpos[site_id] for _, site_id in points
        ]))
    return routes


def _numeric(model: mujoco.MjModel, name: str, expected_size: int) -> np.ndarray:
    numeric_id = _named_id(model, mujoco.mjtObj.mjOBJ_NUMERIC, name)
    size = model.numeric_size[numeric_id]
    if size != expected_size:
        raise ValueError(f"MuJoCo numeric '{name}' must contain {expected_size} values")
    address = model.numeric_adr[numeric_id]
    return model.numeric_data[address:address + size].copy()


def _named_id(model: mujoco.MjModel, object_type: mujoco.mjtObj, name: str) -> int:
    object_id = mujoco.mj_name2id(model, object_type, name)
    if object_id < 0:
        raise ValueError(f"MuJoCo scene is missing required element '{name}'")
    return object_id


def _vector(values: np.ndarray, size: int, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    if vector.shape != (size,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"MuJoCo {name} must contain {size} finite values")
    return vector
