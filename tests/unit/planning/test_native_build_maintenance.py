from pathlib import Path


def test_clean_native_cache_preserves_mpc_artifacts(tmp_path):
    from uav_ac.planning.native.maintenance import clean_native

    generated=tmp_path/'c_generated_code'
    generated.mkdir()
    mpc=generated/'libacados_ocp_solver_quadrotor_wrench_nmpc.so'
    mpc.write_bytes(b'original MPC binary')
    signature=generated/'.quadrotor_wrench_nmpc.signature'
    signature.write_text('MPC cache signature')
    json_file=tmp_path/'quadrotor_wrench_nmpc_ocp.json'
    json_file.write_text('MPC solver configuration')
    native=generated/'cython'/'lib'
    native.mkdir(parents=True)
    (native/'planning_kernel.so').write_bytes(b'rebuildable native output')
    assert clean_native(tmp_path)
    assert not (generated/'cython').exists()
    assert mpc.read_bytes() == b'original MPC binary'
    assert signature.read_text() == 'MPC cache signature'
    assert json_file.read_text() == 'MPC solver configuration'
    assert not clean_native(tmp_path)


def test_build_does_not_report_success_when_optional_kernels_are_missing(tmp_path):
    import pytest
    from uav_ac.planning.native.maintenance import build_native

    (tmp_path/'setup.py').write_text('pass\n')
    generated=tmp_path/'c_generated_code'
    generated.mkdir()
    mpc=generated/'libacados_ocp_solver_quadrotor_wrench_nmpc.so'
    mpc.write_bytes(b'MPC must stay untouched')
    with pytest.raises(RuntimeError, match='did not produce'):
        build_native(tmp_path)
    assert mpc.read_bytes() == b'MPC must stay untouched'
