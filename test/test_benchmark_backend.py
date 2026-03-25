import os

import numpy as np

from dq_nmpc.benchmark_backend import _extract_translation
from dq_nmpc.benchmark_backend import _quaternion_wxyz_to_xyzw
from dq_nmpc.benchmark_backend import odometry_to_state
from dq_nmpc.dq_controller import resolve_acados_paths


def test_resolve_acados_paths_uses_work_dir_defaults(tmp_path):
    json_file, code_export_directory, work_dir = resolve_acados_paths(
        acados_work_dir=tmp_path,
    )

    assert json_file == os.path.join(str(tmp_path), 'acados_ocp_mpc.json')
    assert code_export_directory == os.path.join(str(tmp_path), 'c_generated_code')
    assert work_dir == str(tmp_path)


def test_odometry_to_state_rotates_body_velocity_into_world():
    state = odometry_to_state(
        position=[1.0, 2.0, 3.0],
        quaternion_xyzw=[0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)],
        linear_velocity=[1.0, 0.0, 0.0],
        angular_velocity=[0.1, 0.2, 0.3],
        velocity_frame='body',
    )

    np.testing.assert_allclose(state[0:3], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(state[3:6], [0.0, 1.0, 0.0], atol=1e-8)
    np.testing.assert_allclose(state[6:10], [np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)])
    np.testing.assert_allclose(state[10:13], [0.1, 0.2, 0.3])


def test_quaternion_and_translation_helpers_handle_expected_layouts():
    np.testing.assert_allclose(
        _quaternion_wxyz_to_xyzw([0.5, 0.1, 0.2, 0.3]),
        [0.1, 0.2, 0.3, 0.5],
    )
    np.testing.assert_allclose(_extract_translation([0.0, 4.0, 5.0, 6.0]), [4.0, 5.0, 6.0])
    np.testing.assert_allclose(_extract_translation([4.0, 5.0, 6.0]), [4.0, 5.0, 6.0])
