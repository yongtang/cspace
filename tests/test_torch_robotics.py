import cspace.torch
import cspace.torch.ops

import pathlib
import numpy
import scipy
import torch


def test_rpy_qua(device):
    rpy = (1.5707, 0, -1.5707)
    val = cspace.torch.ops.rpy_to_qua(torch.as_tensor(rpy, device=device))

    qua = scipy.spatial.transform.Rotation.from_euler("xyz", rpy).as_quat()
    assert numpy.allclose(val, qua, atol=1e-4)

    qua = (0.5000, -0.5000, -0.5000, 0.5000)
    assert numpy.allclose(val, qua, atol=1e-4)

    val = cspace.torch.ops.qua_to_rpy(val)
    assert numpy.allclose(val, rpy, atol=1e-4)

    rpy = scipy.spatial.transform.Rotation.from_quat(qua).as_euler("xyz")
    assert numpy.allclose(val, rpy, atol=1e-4)


def test_spec(device, urdf_file):
    spec = cspace.torch.Spec(description=pathlib.Path(urdf_file).read_text())

    joint = spec.joint("base_to_right_leg")

    xyz = joint.origin.xyz
    rpy = joint.origin.rpy
    qua = scipy.spatial.transform.Rotation.from_euler("xyz", rpy).as_quat()

    state = torch.tensor(1.0)

    transform = joint.transform(state)

    assert numpy.allclose(transform.xyz, xyz, atol=1e-4)
    assert numpy.allclose(transform.rpy, rpy, atol=1e-4)
    assert numpy.allclose(transform.qua, qua, atol=1e-4)

    # continuous axis=(0, 1, 0)
    joint = spec.joint("right_front_wheel_joint")

    xyz = joint.origin.xyz
    rpy = (0, 1, 0)
    qua = scipy.spatial.transform.Rotation.from_euler("xyz", rpy).as_quat()

    state = torch.tensor(1.0)

    transform = joint.transform(state)

    assert numpy.allclose(transform.xyz, xyz, atol=1e-4)
    assert numpy.allclose(transform.rpy, rpy, atol=1e-4)
    assert numpy.allclose(transform.qua, qua, atol=1e-4)

    # revolute axis=(0, 0, 1) limit=(0.0, 0.548)
    joint = spec.joint("left_gripper_joint")

    xyz = joint.origin.xyz
    rpy = (0, 0, 0.548)
    qua = scipy.spatial.transform.Rotation.from_euler("xyz", rpy).as_quat()

    state = torch.tensor(1.0)

    transform = joint.transform(state)

    assert numpy.allclose(transform.xyz, xyz, atol=1e-4)
    assert numpy.allclose(transform.rpy, rpy, atol=1e-4)
    assert numpy.allclose(transform.qua, qua, atol=1e-4)

    # prismatic axis=(1, 0, 0) limit=(-0.38, 0)
    joint = spec.joint("gripper_extension")

    xyz = torch.as_tensor(joint.origin.xyz) + torch.as_tensor((-0.38, 0, 0))
    rpy = joint.origin.rpy
    qua = scipy.spatial.transform.Rotation.from_euler("xyz", rpy).as_quat()

    state = torch.tensor(-1.0)

    transform = joint.transform(state)

    assert numpy.allclose(transform.xyz, xyz, atol=1e-4)
    assert numpy.allclose(transform.rpy, rpy, atol=1e-4)
    assert numpy.allclose(transform.qua, qua, atol=1e-4)


def test_kinematics(device, urdf_file):
    spec = cspace.torch.Spec(description=pathlib.Path(urdf_file).read_text())
    kinematics = spec.kinematics("left_gripper")
