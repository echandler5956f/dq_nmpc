import time
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation as R

from .dq_controller import resolve_acados_paths
from .dq_controller import solver as create_solver
from .functions import dualquat_from_pose_casadi
from .ode_acados import dual_velocity_casadi
from .ode_acados import dualquat_quat_casadi
from .ode_acados import dualquat_trans_casadi
from .ode_acados import velocities_from_twist_casadi


_DUALQUAT_FROM_POSE = dualquat_from_pose_casadi()
_DUAL_TWIST = dual_velocity_casadi()
_GET_TRANS = dualquat_trans_casadi()
_GET_QUAT = dualquat_quat_casadi()
_VELOCITY_FROM_TWIST = velocities_from_twist_casadi()


@dataclass
class DQStateSnapshot:
    position: np.ndarray
    linear_velocity_world: np.ndarray
    quaternion_xyzw: np.ndarray
    angular_velocity_body: np.ndarray


@dataclass
class DQSolveSnapshot:
    status: int
    solve_time_ms: float
    control: np.ndarray
    nominal: DQStateSnapshot
    predicted_states: list[DQStateSnapshot]


def _as_array(values, expected_length):
    array = np.asarray(values, dtype=np.double).reshape(-1)
    if array.shape[0] != expected_length:
        raise ValueError(f'Expected vector of length {expected_length}, got {array.shape[0]}.')
    return array


def _extract_translation(dual_quaternion):
    raw = np.asarray(dual_quaternion, dtype=np.double).reshape(-1)
    if raw.shape[0] == 3:
        return raw
    if raw.shape[0] == 4:
        return raw[1:4]

    translation = np.asarray(_GET_TRANS(raw), dtype=np.double).reshape(-1)
    if translation.shape[0] == 4:
        return translation[1:4]
    return _as_array(translation, 3)


def _quaternion_wxyz_to_xyzw(quaternion_wxyz):
    quat = _as_array(quaternion_wxyz, 4)
    return np.array([quat[1], quat[2], quat[3], quat[0]], dtype=np.double)


def odometry_to_state(position, quaternion_xyzw, linear_velocity, angular_velocity, velocity_frame='body'):
    state = np.zeros((13,), dtype=np.double)
    state[0:3] = _as_array(position, 3)

    quat_xyzw = _as_array(quaternion_xyzw, 4)
    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.double)
    state[6:10] = quat_wxyz

    linear = _as_array(linear_velocity, 3)
    if velocity_frame == 'body':
        state[3:6] = R.from_quat(quat_xyzw).as_matrix() @ linear
    elif velocity_frame == 'world':
        state[3:6] = linear
    else:
        raise ValueError(f"Unsupported odometry velocity frame '{velocity_frame}'.")

    state[10:13] = _as_array(angular_velocity, 3)
    return state


def extract_reference_from_points(points, inertia_matrix, horizon_steps):
    point_list = list(points)
    if len(point_list) != horizon_steps:
        raise ValueError(
            f'Expected exactly {horizon_steps} reference points, received {len(point_list)}.'
        )

    x_ref = np.zeros((13, horizon_steps), dtype=np.double)
    u_d = np.zeros((4, horizon_steps), dtype=np.double)
    w_dot_ref = np.zeros((3, horizon_steps), dtype=np.double)
    x_dual = np.zeros((14, horizon_steps), dtype=np.double)

    for index, point in enumerate(point_list):
        position = np.array(
            [point.position.x, point.position.y, point.position.z],
            dtype=np.double,
        )
        velocity = np.array(
            [point.velocity.x, point.velocity.y, point.velocity.z],
            dtype=np.double,
        )
        quaternion_wxyz = np.array(
            [point.quaternion.w, point.quaternion.x, point.quaternion.y, point.quaternion.z],
            dtype=np.double,
        )
        angular_velocity = np.array(
            [point.angular_velocity.x, point.angular_velocity.y, point.angular_velocity.z],
            dtype=np.double,
        )
        angular_acceleration = np.array(
            [
                point.angular_velocity_dot.x,
                point.angular_velocity_dot.y,
                point.angular_velocity_dot.z,
            ],
            dtype=np.double,
        )

        x_ref[0:3, index] = position
        x_ref[3:6, index] = velocity
        x_ref[6:10, index] = quaternion_wxyz
        x_ref[10:13, index] = angular_velocity

        dual_pose = _DUALQUAT_FROM_POSE(
            quaternion_wxyz[0],
            quaternion_wxyz[1],
            quaternion_wxyz[2],
            quaternion_wxyz[3],
            position[0],
            position[1],
            position[2],
        )
        dual_twist = _DUAL_TWIST(
            np.array(
                [
                    angular_velocity[0],
                    angular_velocity[1],
                    angular_velocity[2],
                    velocity[0],
                    velocity[1],
                    velocity[2],
                ],
                dtype=np.double,
            ),
            dual_pose,
        )

        x_dual[0:8, index] = np.asarray(dual_pose, dtype=np.double).reshape(8,)
        x_dual[8:14, index] = np.asarray(dual_twist, dtype=np.double).reshape(6,)

        u_d[0, index] = float(point.force)
        w_dot_ref[:, index] = angular_acceleration
        u_d[1:4, index] = inertia_matrix @ angular_acceleration + np.cross(
            angular_velocity,
            inertia_matrix @ angular_velocity,
        )

    return x_ref, u_d, w_dot_ref, x_dual


class DQBenchmarkCore:
    def __init__(
        self,
        params,
        build=False,
        json_file=None,
        acados_work_dir=None,
        code_export_directory=None,
        verbose=True,
    ):
        self.params = params
        self.mass = float(params['mass'])
        self.gravity = float(params['gravity'])
        self.inertia_matrix = np.array(
            [
                [params['ixx'], 0.0, 0.0],
                [0.0, params['iyy'], 0.0],
                [0.0, 0.0, params['izz']],
            ],
            dtype=np.double,
        )

        self.horizon_steps = int(params['nmpc']['horizon_steps'])
        self.horizon_time = float(params['nmpc']['horizon_time'])
        self.ts = float(params['nmpc']['ts'])
        self.q_weights = np.asarray(params['nmpc']['Q'], dtype=np.double)
        self.qe_weights = np.asarray(params['nmpc']['Q_e'], dtype=np.double)
        self.r_weights = np.asarray(params['nmpc']['R'], dtype=np.double)

        self.json_file, self.code_export_directory, self.acados_work_dir = resolve_acados_paths(
            json_file=json_file,
            acados_work_dir=acados_work_dir,
            code_export_directory=code_export_directory,
        )

        self.acados_ocp_solver, self.ocp = create_solver(
            params,
            flag=build,
            json_file=self.json_file,
            acados_work_dir=self.acados_work_dir,
            code_export_directory=self.code_export_directory,
            verbose=verbose,
        )

        self.current_state = np.zeros((13,), dtype=np.double)
        self.current_dual_state = np.zeros((14,), dtype=np.double)
        self.current_dual_state[0] = 1.0

        self.reference_state = np.zeros((14, self.horizon_steps), dtype=np.double)
        self.reference_input = np.zeros((4, self.horizon_steps), dtype=np.double)

        self.has_odometry = False
        self.has_reference = False
        self.last_status = None
        self.last_solve_time_ms = 0.0

    def set_odometry(
        self,
        position,
        quaternion_xyzw,
        linear_velocity,
        angular_velocity,
        velocity_frame='body',
    ):
        self.current_state = odometry_to_state(
            position,
            quaternion_xyzw,
            linear_velocity,
            angular_velocity,
            velocity_frame=velocity_frame,
        )

        dual_pose = _DUALQUAT_FROM_POSE(
            self.current_state[6],
            self.current_state[7],
            self.current_state[8],
            self.current_state[9],
            self.current_state[0],
            self.current_state[1],
            self.current_state[2],
        )
        dual_twist = _DUAL_TWIST(
            np.array(
                [
                    self.current_state[10],
                    self.current_state[11],
                    self.current_state[12],
                    self.current_state[3],
                    self.current_state[4],
                    self.current_state[5],
                ],
                dtype=np.double,
            ),
            dual_pose,
        )

        self.current_dual_state[0:8] = np.asarray(dual_pose, dtype=np.double).reshape(8,)
        self.current_dual_state[8:14] = np.asarray(dual_twist, dtype=np.double).reshape(6,)
        self.has_odometry = True
        return self.current_state.copy()

    def set_odometry_from_message(self, msg, velocity_frame='body'):
        return self.set_odometry(
            position=[
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
                msg.pose.pose.position.z,
            ],
            quaternion_xyzw=[
                msg.pose.pose.orientation.x,
                msg.pose.pose.orientation.y,
                msg.pose.pose.orientation.z,
                msg.pose.pose.orientation.w,
            ],
            linear_velocity=[
                msg.twist.twist.linear.x,
                msg.twist.twist.linear.y,
                msg.twist.twist.linear.z,
            ],
            angular_velocity=[
                msg.twist.twist.angular.x,
                msg.twist.twist.angular.y,
                msg.twist.twist.angular.z,
            ],
            velocity_frame=velocity_frame,
        )

    def set_reference_from_points(self, points):
        _, self.reference_input, _, self.reference_state = extract_reference_from_points(
            points,
            self.inertia_matrix,
            self.horizon_steps,
        )
        self.has_reference = True

    def set_reference_from_message(self, msg):
        self.set_reference_from_points(msg.points)

    def clear_odometry(self):
        self.has_odometry = False

    def clear_reference(self):
        self.has_reference = False

    def ready(self):
        return self.has_odometry and self.has_reference

    def decode_dual_state(self, dual_state):
        dual_state = np.asarray(dual_state, dtype=np.double).reshape(-1)
        if dual_state.shape[0] != 14:
            raise ValueError(f'Expected 14D dual state, got {dual_state.shape[0]}.')

        dual_pose = dual_state[0:8]
        dual_twist = dual_state[8:14]

        quaternion_wxyz = np.asarray(_GET_QUAT(dual_pose), dtype=np.double).reshape(-1)
        velocities = np.asarray(
            _VELOCITY_FROM_TWIST(dual_twist, dual_pose),
            dtype=np.double,
        ).reshape(-1)

        return DQStateSnapshot(
            position=_extract_translation(dual_pose),
            linear_velocity_world=velocities[3:6].copy(),
            quaternion_xyzw=_quaternion_wxyz_to_xyzw(quaternion_wxyz),
            angular_velocity_body=velocities[0:3].copy(),
        )

    def solve(self):
        if not self.ready():
            raise RuntimeError('DQBenchmarkCore requires both odometry and reference before solve().')

        self.acados_ocp_solver.set(0, 'lbx', self.current_dual_state)
        self.acados_ocp_solver.set(0, 'ubx', self.current_dual_state)

        for stage in range(self.horizon_steps):
            parameters = np.hstack(
                (
                    self.reference_state[:, stage],
                    self.reference_input[:, stage],
                    self.q_weights,
                    self.qe_weights,
                    self.r_weights,
                )
            )
            self.acados_ocp_solver.set(stage, 'p', parameters)

        self.acados_ocp_solver.set(self.horizon_steps, 'p', parameters)

        start = time.perf_counter()
        self.last_status = int(self.acados_ocp_solver.solve())
        self.last_solve_time_ms = (time.perf_counter() - start) * 1000.0

        control = np.asarray(self.acados_ocp_solver.get(0, 'u'), dtype=np.double).reshape(4,)
        nominal_state = np.asarray(self.acados_ocp_solver.get(1, 'x'), dtype=np.double).reshape(14,)
        predicted_states = [
            self.decode_dual_state(
                np.asarray(self.acados_ocp_solver.get(stage, 'x'), dtype=np.double).reshape(14,)
            )
            for stage in range(1, self.horizon_steps + 1)
        ]

        return DQSolveSnapshot(
            status=self.last_status,
            solve_time_ms=self.last_solve_time_ms,
            control=control,
            nominal=self.decode_dual_state(nominal_state),
            predicted_states=predicted_states,
        )
