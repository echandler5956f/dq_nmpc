import copy
import math
import time
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation as R

from .dq_controller import resolve_acados_paths
from .dq_controller import solver as create_solver

DQ_STATE_DIM = 14
DQ_CONTROL_DIM = 4


def normalize_benchmark_params(params):
    """Adapt the benchmark schema to the legacy solver schema.

    The standalone dq-nmpc pipeline still uses its historical top-level
    ``mass``/``ixx``/``nmpc`` structure.  LissajousTests supplies a smaller
    benchmark schema where vehicle and transport settings are shared with
    SRE.  Keep the translation at this boundary so the lab-facing solver API
    remains unchanged.
    """
    if 'vehicle' not in params or 'dq_nmpc' not in params:
        return copy.deepcopy(params)

    vehicle = params['vehicle']
    cost = params['cost']
    dq = params['dq_nmpc']
    inertia = vehicle['inertia']
    limits = vehicle['control_limits']
    position = float(cost['position'])
    orientation = float(cost['orientation'])
    linear_velocity = float(cost['linear_velocity'])
    angular_velocity = float(cost['angular_velocity'])
    thrust = float(cost['control_effort']['thrust'])
    moment = float(cost['control_effort']['moment'])
    return {
        # The benchmark has one fixed DQ model; this is an artifact label,
        # not a LissajousTests tuning parameter.
        'mav_name': 'quadrotor',
        'mass': float(vehicle['mass']),
        'gravity': float(vehicle['gravity']),
        'ixx': float(inertia['xx']),
        'iyy': float(inertia['yy']),
        'izz': float(inertia['zz']),
        'drag_linear': [
            float(vehicle['drag']['linear'][axis]) for axis in ('x', 'y', 'z')
        ],
        'drag_quadratic': [
            float(vehicle['drag']['quadratic'][axis]) for axis in ('x', 'y', 'z')
        ],
        'thrust_axis_body': [
            float(vehicle['thrust_axis_body'][axis]) for axis in ('x', 'y', 'z')
        ],
        'nmpc': {
            # The scalar slots of each logarithmic quaternion are identically
            # zero.  The six benchmark weights therefore expand only over the
            # physical vector components; axis-specific tuning is unsupported.
            'Q': [
                0.0, position, position, position,
                linear_velocity, linear_velocity, linear_velocity,
                0.0, orientation, orientation, orientation,
                angular_velocity, angular_velocity, angular_velocity,
            ],
            'Q_e': [
                0.0, position, position, position,
                linear_velocity, linear_velocity, linear_velocity,
                0.0, orientation, orientation, orientation,
                angular_velocity, angular_velocity, angular_velocity,
            ],
            'R': [thrust, moment, moment, moment],
            # The dual-quaternion model has fixed dimensions.  These are
            # solver internals, not tuning parameters.
            'nx': DQ_STATE_DIM,
            'nu': DQ_CONTROL_DIM,
            'lbu': [
                float(limits['thrust_min']),
                float(limits['mx_min']),
                float(limits['my_min']),
                float(limits['mz_min']),
            ],
            'ubu': [
                float(limits['thrust_max']),
                float(limits['mx_max']),
                float(limits['my_max']),
                float(limits['mz_max']),
            ],
            'horizon_steps': int(dq['horizon_steps']),
            'horizon_time': float(dq['horizon_time']),
            'integrator_type': str(dq.get('integrator_type', 'IRK')).upper(),
            'integrator_stages': int(dq.get('integrator_stages', 4)),
            'integrator_newton_iterations': int(dq.get('integrator_newton_iterations', 2)),
            'levenberg_marquardt': float(dq.get('levenberg_marquardt', 10.0)),
        },
    }


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


def _as_array(values, expected_length):
    array = np.asarray(values, dtype=np.double).reshape(-1)
    if array.shape[0] != expected_length:
        raise ValueError(f'Expected vector of length {expected_length}, got {array.shape[0]}.')
    return array

# These raw numpy quaternion/transform operators are added here for performance reasons. The existing casadi callbacks cause nontrivial slowdown.
def _quaternion_multiply(first, second):
    return np.array(
        [
            first[0] * second[0] - np.dot(first[1:], second[1:]),
            *(first[0] * second[1:] + second[0] * first[1:] + np.cross(first[1:], second[1:])),
        ],
        dtype=np.double,
    )


def _dual_quaternion_from_pose(quaternion_wxyz, position):
    real = _as_array(quaternion_wxyz, 4)
    translation = _as_array(position, 3)
    dual = 0.5 * _quaternion_multiply(np.r_[0.0, translation], real)
    return np.concatenate((real, dual))


def _dual_twist_from_world_velocity(quaternion_wxyz, angular_velocity, linear_velocity):
    quat_xyzw = _quaternion_wxyz_to_xyzw(quaternion_wxyz)
    angular = _as_array(angular_velocity, 3)
    linear = R.from_quat(quat_xyzw).inv().apply(_as_array(linear_velocity, 3))
    return np.concatenate((angular, linear))


def _translation_from_dual_quaternion(dual_quaternion):
    raw = _as_array(dual_quaternion, 8)
    real = raw[0:4]
    dual = raw[4:8]
    return (2.0 * _quaternion_multiply(dual, np.r_[real[0], -real[1:]]))[1:4]


def _extract_translation(dual_quaternion):
    raw = np.asarray(dual_quaternion, dtype=np.double).reshape(-1)
    if raw.shape[0] == 3:
        return raw
    if raw.shape[0] == 4:
        return raw[1:4]

    if raw.shape[0] == 8:
        return _translation_from_dual_quaternion(raw)
    raise ValueError(f'Expected translation, quaternion, or dual quaternion; got {raw.shape[0]} values.')


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

    reference_input = np.zeros((4, horizon_steps), dtype=np.double)
    reference_state = np.zeros((14, horizon_steps), dtype=np.double)
    ixx, iyy, izz = np.diag(inertia_matrix)

    for index, point in enumerate(point_list):
        px, py, pz = point.position.x, point.position.y, point.position.z
        vx, vy, vz = point.velocity.x, point.velocity.y, point.velocity.z
        qw, qx, qy, qz = (
            point.quaternion.w,
            point.quaternion.x,
            point.quaternion.y,
            point.quaternion.z,
        )
        wx, wy, wz = (
            point.angular_velocity.x,
            point.angular_velocity.y,
            point.angular_velocity.z,
        )
        ax, ay, az = (
            point.angular_velocity_dot.x,
            point.angular_velocity_dot.y,
            point.angular_velocity_dot.z,
        )

        reference_state[0:4, index] = qw, qx, qy, qz
        reference_state[4:8, index] = (
            -0.5 * (px * qx + py * qy + pz * qz),
            0.5 * (qw * px + py * qz - pz * qy),
            0.5 * (qw * py + pz * qx - px * qz),
            0.5 * (qw * pz + px * qy - py * qx),
        )
        inverse_norm = 1.0 / math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
        qw, qx, qy, qz = (
            qw * inverse_norm,
            qx * inverse_norm,
            qy * inverse_norm,
            qz * inverse_norm,
        )
        reference_state[8:11, index] = wx, wy, wz
        reference_state[11:14, index] = (
            (1.0 - 2.0 * (qy * qy + qz * qz)) * vx
            + 2.0 * (qx * qy + qw * qz) * vy
            + 2.0 * (qx * qz - qw * qy) * vz,
            2.0 * (qx * qy - qw * qz) * vx
            + (1.0 - 2.0 * (qx * qx + qz * qz)) * vy
            + 2.0 * (qy * qz + qw * qx) * vz,
            2.0 * (qx * qz + qw * qy) * vx
            + 2.0 * (qy * qz - qw * qx) * vy
            + (1.0 - 2.0 * (qx * qx + qy * qy)) * vz,
        )
        reference_input[0, index] = float(point.force)
        reference_input[1:4, index] = (
            ixx * ax + (izz - iyy) * wy * wz,
            iyy * ay + (ixx - izz) * wx * wz,
            izz * az + (iyy - ixx) * wx * wy,
        )

    return reference_input, reference_state


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
        self.params = normalize_benchmark_params(params)
        params = self.params
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
        self.integrator_type = str(params['nmpc'].get('integrator_type', 'IRK')).upper()
        self.integrator_stages = int(params['nmpc'].get('integrator_stages', 4))
        self.integrator_newton_iterations = int(
            params['nmpc'].get('integrator_newton_iterations', 2)
        )
        self.levenberg_marquardt = float(params['nmpc'].get('levenberg_marquardt', 10.0))
        # ``ts`` belongs to the historical standalone API.  The benchmark
        # adapter intentionally omits it: Acados uses horizon_time /
        # horizon_steps for the active prediction interval.
        self.ts = float(params['nmpc'].get('ts', 0.0))
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
        self.stage_parameters = np.empty((self.horizon_steps, 50), dtype=np.double)
        self.stage_parameters[:, 18:32] = self.q_weights
        self.stage_parameters[:, 32:46] = self.qe_weights
        self.stage_parameters[:, 46:50] = self.r_weights

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

        dual_pose = _dual_quaternion_from_pose(self.current_state[6:10], self.current_state[0:3])
        dual_twist = _dual_twist_from_world_velocity(
            self.current_state[6:10],
            self.current_state[10:13],
            self.current_state[3:6],
        )

        self.current_dual_state[0:8] = dual_pose
        self.current_dual_state[8:14] = dual_twist
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
        self.reference_input, self.reference_state = extract_reference_from_points(
            points,
            self.inertia_matrix,
            self.horizon_steps,
        )
        self.stage_parameters[:, 0:14] = self.reference_state.T
        self.stage_parameters[:, 14:18] = self.reference_input.T
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

        quaternion_wxyz = dual_pose[0:4]
        velocities = np.concatenate(
            (
                dual_twist[0:3],
                R.from_quat(_quaternion_wxyz_to_xyzw(quaternion_wxyz)).apply(dual_twist[3:6]),
            )
        )

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
            self.acados_ocp_solver.set(stage, 'p', self.stage_parameters[stage])

        self.acados_ocp_solver.set(self.horizon_steps, 'p', self.stage_parameters[-1])

        start = time.perf_counter()
        self.last_status = int(self.acados_ocp_solver.solve())
        self.last_solve_time_ms = (time.perf_counter() - start) * 1000.0

        control = np.asarray(self.acados_ocp_solver.get(0, 'u'), dtype=np.double).reshape(4,)
        nominal_state = np.asarray(self.acados_ocp_solver.get(1, 'x'), dtype=np.double).reshape(14,)

        return DQSolveSnapshot(
            status=self.last_status,
            solve_time_ms=self.last_solve_time_ms,
            control=control,
            nominal=self.decode_dual_state(nominal_state),
        )
