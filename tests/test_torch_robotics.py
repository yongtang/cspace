import pytest

import cspace.torch
import cspace.torch.ops

import transforms3d
import pathlib
import numpy
import scipy
import torch


@pytest.mark.parametrize("angle_r", list(range(-180, 180, 30)))
@pytest.mark.parametrize("angle_p", list(range(-180, 180, 30)))
@pytest.mark.parametrize("angle_y", list(range(-180, 180, 30)))
def test_ops(angle_r, angle_p, angle_y, device):
    def t3d_rpy_to_qua(rpy):
        qw, qx, qy, qz = transforms3d.euler.euler2quat(rpy[0], rpy[1], rpy[2])
        return [qx, qy, qz, qw]

    def t3d_qua_to_rot(qua):
        qx, qy, qz, qw = qua
        return transforms3d.euler.quat2mat([qw, qx, qy, qz])

    def t3d_rot_to_rpy(rot):
        return transforms3d.euler.mat2euler(rot)

    def t3d_qua_to_rpy(qua):
        qx, qy, qz, qw = qua
        return transforms3d.euler.quat2euler([qw, qx, qy, qz])

    def t3d_rpy_to_rot(rpy):
        return transforms3d.euler.euler2mat(rpy[0], rpy[1], rpy[2])

    def t3d_rot_to_rpy(rot):
        return transforms3d.euler.mat2euler(rot)

    def t3d_rot_to_qua(rot):
        rot = numpy.array(rot)
        qw, qx, qy, qz = transforms3d.quaternions.mat2quat(rot)
        return [qx, qy, qz, qw]

    rpy = scipy.special.radian((angle_r, angle_p, angle_y), 0, 0)
    rpy = torch.as_tensor(rpy, dtype=torch.float64)

    rot = cspace.torch.ops.rpy_to_rot(rpy)
    assert numpy.allclose(rot, t3d_rpy_to_rot(rpy), atol=1e-4)

    val = cspace.torch.ops.rot_to_rpy(rot)
    assert numpy.allclose(val, t3d_rot_to_rpy(rot), atol=1e-4)

    qua = cspace.torch.ops.rpy_to_qua(rpy)
    assert numpy.allclose(qua, t3d_rpy_to_qua(rpy), atol=1e-4)

    rot = cspace.torch.ops.qua_to_rot(qua)
    assert numpy.allclose(rot, t3d_qua_to_rot(qua), atol=1e-4)

    qua = cspace.torch.ops.rot_to_qua(rot)
    assert numpy.allclose(qua, t3d_rot_to_qua(rot), atol=1e-4)


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
            assert numpy.allclose(true, data, atol=1e-4)


def test_kinematics(device, urdf_file, joint_state, link_pose):
    spec = cspace.torch.Spec(description=pathlib.Path(urdf_file).read_text())
    kinematics = spec.kinematics("left_gripper")

    joint_state = dict(zip(joint_state.name, joint_state.position))
    state = [joint_state.get(joint.name, 0.0) for joint in spec.joint]
    state = torch.as_tensor(state, dtype=torch.float64)
    true = torch.tensor(
        [
            link_pose["left_gripper"].pose.position.x,
            link_pose["left_gripper"].pose.position.y,
            link_pose["left_gripper"].pose.position.z,
            link_pose["left_gripper"].pose.orientation.x,
            link_pose["left_gripper"].pose.orientation.y,
            link_pose["left_gripper"].pose.orientation.z,
            link_pose["left_gripper"].pose.orientation.w,
        ]
    ).unsqueeze(-2)

    pose = kinematics.forward(state)
    assert pose.shape == true.shape
    assert numpy.allclose(pose, true, atol=1e-4)
