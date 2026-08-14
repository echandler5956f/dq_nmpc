import os
import sys
from copy import deepcopy

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import dq_nmpc.dq_controller as dq_controller
from dq_nmpc.benchmark_backend import _extract_translation
from dq_nmpc.benchmark_backend import _quaternion_wxyz_to_xyzw
from dq_nmpc.benchmark_backend import DQBenchmarkCore
from dq_nmpc.benchmark_backend import normalize_benchmark_params
from dq_nmpc.benchmark_backend import odometry_to_state
from dq_nmpc.dq_controller import _acados_signature_file
from dq_nmpc.dq_controller import _acados_solver_artifacts_match
from dq_nmpc.dq_controller import _shared_lib_name
from dq_nmpc.dq_controller import _solver_generation_signature
from dq_nmpc.dq_controller import _save_json
from dq_nmpc.dq_controller import resolve_acados_paths
from dq_nmpc.utils import yaml_to_dict


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


def test_clear_methods_drop_readiness_flags_without_constructing_solver():
    core = object.__new__(DQBenchmarkCore)
    core.has_odometry = True
    core.has_reference = True

    core.clear_odometry()
    assert not core.has_odometry
    assert core.has_reference

    core.clear_reference()
    assert not core.has_reference


def test_benchmark_schema_expands_shared_uniform_cost_weights():
    params = {
        'vehicle': {
            'mass': 0.875,
            'gravity': 9.80665,
            'inertia': {'xx': 1.0, 'yy': 2.0, 'zz': 3.0},
            'drag': {
                'linear': {'x': 0.1, 'y': 0.2, 'z': 0.3},
                'quadratic': {'x': 0.4, 'y': 0.5, 'z': 0.6},
            },
            'thrust_axis_body': {'x': 0.0, 'y': 0.0, 'z': 1.0},
            'control_limits': {
                'thrust_min': 0.0,
                'thrust_max': 60.0,
                'mx_min': -2.0,
                'mx_max': 2.0,
                'my_min': -2.0,
                'my_max': 2.0,
                'mz_min': -0.3,
                'mz_max': 0.3,
            },
        },
        'cost': {
            'position': 210.0,
            'orientation': 5.0,
            'linear_velocity': 1.0,
            'angular_velocity': 2.0,
            'control_effort': {'thrust': 0.5, 'moment': 0.1},
        },
        'dq_nmpc': {'horizon_steps': 21, 'horizon_time': 1.0},
    }

    normalized = normalize_benchmark_params(params)

    assert normalized['nmpc']['Q'] == [
        0.0, 210.0, 210.0, 210.0,
        1.0, 1.0, 1.0,
        0.0, 5.0, 5.0, 5.0,
        2.0, 2.0, 2.0,
    ]
    assert normalized['nmpc']['Q_e'] == normalized['nmpc']['Q']
    assert normalized['nmpc']['R'] == [0.5, 0.1, 0.1, 0.1]
    assert normalized['nmpc']['horizon_time'] == 1.0


def test_solver_generation_signature_ignores_runtime_only_tuning():
    params = yaml_to_dict('config/mujoco/default/dq_control.yaml')
    updated = deepcopy(params)

    updated['nmpc']['Q'][0] += 1.0
    updated['nmpc']['Q_e'][0] += 1.0
    updated['nmpc']['R'][0] += 0.1
    updated['nmpc']['ts'] = 0.5

    assert _solver_generation_signature(updated) == _solver_generation_signature(params)


def test_solver_generation_signature_changes_for_codegen_inputs():
    params = yaml_to_dict('config/mujoco/default/dq_control.yaml')
    updated = deepcopy(params)
    updated['mass'] += 0.1

    assert _solver_generation_signature(updated) != _solver_generation_signature(params)


def test_solver_generation_signature_records_source_hash():
    params = yaml_to_dict('config/mujoco/default/dq_control.yaml')
    signature = _solver_generation_signature(params)

    assert signature['version'] == 2
    assert signature['solver_source_sha256'] == dq_controller._solver_source_sha256()
    assert len(signature['solver_source_sha256']) == 64
    int(signature['solver_source_sha256'], 16)


def test_solver_generation_signature_changes_with_source_hash(monkeypatch):
    params = yaml_to_dict('config/mujoco/default/dq_control.yaml')
    original = _solver_generation_signature(params)

    monkeypatch.setattr(dq_controller, '_solver_source_sha256', lambda: '0' * 64)
    changed = _solver_generation_signature(params)

    assert changed != original
    assert changed['solver_source_sha256'] == '0' * 64


def test_acados_solver_artifacts_match_uses_signature_and_paths(tmp_path):
    params = yaml_to_dict('config/mujoco/default/dq_control.yaml')
    json_file, code_export_directory, _ = resolve_acados_paths(acados_work_dir=tmp_path)

    os.makedirs(code_export_directory, exist_ok=True)

    _save_json(
        json_file,
        {
            'name': params['mav_name'],
            'code_gen_opts': {'code_export_directory': code_export_directory},
        },
    )
    _save_json(
        _acados_signature_file(json_file),
        _solver_generation_signature(params),
    )

    shared_lib = os.path.join(code_export_directory, _shared_lib_name(params['mav_name']))
    with open(shared_lib, 'w', encoding='utf-8') as stream:
        stream.write('')

    assert _acados_solver_artifacts_match(params, json_file, code_export_directory)

    mismatch = deepcopy(params)
    mismatch['nmpc']['horizon_steps'] += 1
    assert not _acados_solver_artifacts_match(mismatch, json_file, code_export_directory)
