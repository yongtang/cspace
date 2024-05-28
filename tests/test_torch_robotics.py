import cspace.torch
import cspace.torch.ops

import pathlib
import numpy
import scipy
import torch


def test_ops(device):
    val = torch.as_tensor((0.5000, -0.5000, -0.5000, 0.5000), device=device)
    sci = scipy.spatial.transform.Rotation.from_quat(val)

    rpy = cspace.torch.ops.qua_to_rpy(val)
    assert numpy.allclose(rpy, sci.as_euler("xyz"), atol=1e-4)
    assert numpy.allclose(rpy, (1.5707, 0, -1.5707), atol=1e-4)

    qua = cspace.torch.ops.rpy_to_qua(rpy)
    assert numpy.allclose(qua, sci.as_quat(), atol=1e-4)
    assert numpy.allclose(qua, (0.5000, -0.5000, -0.5000, 0.5000), atol=1e-4)

    rot = cspace.torch.ops.qua_to_rot(qua)
    assert numpy.allclose(rot, sci.as_matrix(), atol=1e-4)
    assert numpy.allclose(rot, [[0, 0, -1], [-1, 0, 0], [0, 1, 0]], atol=1e-4)

    qua = cspace.torch.ops.rot_to_qua(rot)
    assert numpy.allclose(qua, sci.as_quat(), atol=1e-4)
    assert numpy.allclose(qua, (0.5000, -0.5000, -0.5000, 0.5000), atol=1e-4)


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


def test_transform(device, urdf_file, joint_state, link_pose):
    spec = cspace.torch.Spec(description=pathlib.Path(urdf_file).read_text())

    joint_state = dict(zip(joint_state.name, joint_state.position))

    state = [joint_state.get(joint.name, 0.0) for joint in spec.joint]
    state = torch.as_tensor(state, dtype=torch.float64)

    for chain in spec.chain:
        link = chain[-1]
        if link in link_pose:
            data = spec.forward(state, link)
            true = torch.tensor(
                [
                    link_pose[link].pose.position.x,
                    link_pose[link].pose.position.y,
                    link_pose[link].pose.position.z,
                    link_pose[link].pose.orientation.x,
                    link_pose[link].pose.orientation.y,
                    link_pose[link].pose.orientation.z,
                    link_pose[link].pose.orientation.w,
                ]
            ).unsqueeze(-2)
            assert numpy.allclose(true, data, atol=5e-2)


def test_kinematics(device, urdf_file):
    spec = cspace.torch.Spec(description=pathlib.Path(urdf_file).read_text())
    kinematics = spec.kinematics("left_gripper")
