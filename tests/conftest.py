import pytest

import argparse
import requests
import operator
import functools
import collections
import transforms3d
import scipy
import numpy
import torch


def pytest_addoption(parser):
    parser.addoption(
        "--train",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="run tests for training",
    )
    parser.addoption(
        "--device", action="store", default="cpu,cuda", help="run tests with CUDA"
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--train"):
        return
    for item in items:
        if "train" in item.keywords:
            item.add_marker(pytest.mark.skip(reason="--no-train"))


def pytest_generate_tests(metafunc):
    device = metafunc.config.option.device.split(",")
    if "device" in metafunc.fixturenames:
        metafunc.parametrize("device", list(map(torch.device, device)), ids=device)


@pytest.fixture(scope="session")
def transforms3d_data(request):
    def f_rpy(euler):
        return numpy.array(euler)

    def f_qua(euler):
        qw, qx, qy, qz = transforms3d.euler.euler2quat(euler[0], euler[1], euler[2])
        return numpy.array((qx, qy, qz, qw))

    def f_rot(euler):
        return numpy.array(transforms3d.euler.euler2mat(euler[0], euler[1], euler[2]))

    def f_rpy_to_rot(euler):
        return numpy.array(transforms3d.euler.euler2mat(euler[0], euler[1], euler[2]))

    def f_rot_to_rpy(euler):
        return numpy.array(
            transforms3d.euler.mat2euler(
                transforms3d.euler.euler2mat(euler[0], euler[1], euler[2])
            )
        )

    def f_rpy_to_qua(euler):
        qw, qx, qy, qz = transforms3d.euler.euler2quat(euler[0], euler[1], euler[2])
        return numpy.array((qx, qy, qz, qw))

    def f_qua_to_rot(euler):
        return numpy.array(
            transforms3d.euler.quat2mat(
                transforms3d.euler.euler2quat(euler[0], euler[1], euler[2])
            )
        )

    def f_rot_to_qua(euler):
        qw, qx, qy, qz = transforms3d.quaternions.mat2quat(
            transforms3d.euler.euler2mat(euler[0], euler[1], euler[2])
        )
        return numpy.array((qx, qy, qz, qw))

    def f_so3_log(euler):
        axis, angle = transforms3d.axangles.mat2axangle(
            transforms3d.euler.euler2mat(euler[0], euler[1], euler[2])
        )
        return numpy.array(axis * angle)

    def f_se3_log(euler, linear):
        t = numpy.expand_dims(numpy.array(linear), axis=-1)
        r = numpy.array(transforms3d.euler.euler2mat(euler[0], euler[1], euler[2]))
        m = numpy.concatenate(
            (numpy.concatenate((r, t), axis=-1), numpy.array([[0, 0, 0, 1]]))
        )
        angle = numpy.abs(numpy.arccos((numpy.trace(r) - 1.0) / 2.0))
        if numpy.abs(angle - numpy.pi) <= numpy.finfo(r.dtype).eps:
            return numpy.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        m = scipy.linalg.logm(m)
        t = m[:3, 3:]
        t = numpy.squeeze(t, axis=-1)

        r = m[:3, :3]
        rx = (r[2, 1] - r[1, 2]) / 2
        ry = (r[0, 2] - r[2, 0]) / 2
        rz = (r[1, 0] - r[0, 1]) / 2
        r = numpy.array([rx, ry, rz])

        return numpy.concatenate((t, r))

    def f_se3_mul(euler, linear):
        t = numpy.expand_dims(numpy.array(linear), axis=-1)
        r = numpy.array(transforms3d.euler.euler2mat(euler[0], euler[1], euler[2]))
        m = numpy.concatenate(
            (numpy.concatenate((r, t), axis=-1), numpy.array([[0, 0, 0, 1]]))
        )
        m = numpy.matrix(m)
        m = m * m
        m = numpy.array(m)
        return m[:3, :]

    def f_se3_inv(euler, linear):
        t = numpy.expand_dims(numpy.array(linear), axis=-1)
        r = numpy.array(transforms3d.euler.euler2mat(euler[0], euler[1], euler[2]))
        m = numpy.concatenate(
            (numpy.concatenate((r, t), axis=-1), numpy.array([[0, 0, 0, 1]]))
        )
        m = numpy.array(scipy.linalg.inv(m))
        return m[:3, :]

    def f_se3_xyz(euler, linear):
        return numpy.array(linear)

    def f_euler(angle_r, angle_p, angle_y, interleave):
        return scipy.special.radian(
            (angle_r + interleave, angle_p + interleave, angle_y + interleave), 0, 0
        )

    def f_linear(linear, interleave):
        return [float(e + interleave) for e in linear]

    def f_so3_entries(f_t3d, angle_r, angle_p, angle_y, batch, interleave):
        count = functools.reduce(operator.mul, batch, 1)
        entries = tuple(
            map(
                lambda index: f_t3d(
                    f_euler(angle_r, angle_p, angle_y, interleave * index)
                ),
                range(count),
            )
        )
        return (
            numpy.reshape(
                numpy.stack(entries), tuple(batch) + next(iter(entries)).shape
            )
            if len(batch)
            else next(iter(entries))
        )

    def f_se3_entries(f_se3, angle_r, angle_p, angle_y, linear, batch, interleave):
        count = functools.reduce(operator.mul, batch, 1)
        entries = tuple(
            map(
                lambda index: f_se3(
                    f_euler(angle_r, angle_p, angle_y, interleave * index),
                    f_linear(linear, interleave * index),
                ),
                range(count),
            )
        )
        return (
            numpy.reshape(
                numpy.stack(entries), tuple(batch) + next(iter(entries)).shape
            )
            if len(batch)
            else next(iter(entries))
        )

    angle_r, angle_p, angle_y, linear, batch, interleave = request.param
    entries = (
        f_so3_entries(f_rpy, angle_r, angle_p, angle_y, batch, interleave),
        f_so3_entries(f_qua, angle_r, angle_p, angle_y, batch, interleave),
        f_so3_entries(f_rot, angle_r, angle_p, angle_y, batch, interleave),
        f_so3_entries(f_rpy_to_rot, angle_r, angle_p, angle_y, batch, interleave),
        f_so3_entries(f_rot_to_rpy, angle_r, angle_p, angle_y, batch, interleave),
        f_so3_entries(f_rpy_to_qua, angle_r, angle_p, angle_y, batch, interleave),
        f_so3_entries(f_qua_to_rot, angle_r, angle_p, angle_y, batch, interleave),
        f_so3_entries(f_rot_to_qua, angle_r, angle_p, angle_y, batch, interleave),
        f_so3_entries(f_so3_log, angle_r, angle_p, angle_y, batch, interleave),
        f_se3_entries(f_se3_log, angle_r, angle_p, angle_y, linear, batch, interleave),
        f_se3_entries(f_se3_mul, angle_r, angle_p, angle_y, linear, batch, interleave),
        f_se3_entries(f_se3_inv, angle_r, angle_p, angle_y, linear, batch, interleave),
        f_se3_entries(f_se3_xyz, angle_r, angle_p, angle_y, linear, batch, interleave),
        batch,
    )

    return entries


@pytest.fixture(scope="session")
def urdf_file_tutorial(tmp_path_factory):
    url = (
        "https://raw.githubusercontent.com/ros/urdf_tutorial/ros2/urdf/07-physics.urdf"
    )
    file = tmp_path_factory.mktemp("data") / "07-physics.urdf"
    file.write_bytes(requests.get(url).content)
    return file


@pytest.fixture(scope="session")
def joint_state_tutorial(urdf_file_tutorial):
    name = [
        "right_front_wheel_joint",
        "right_back_wheel_joint",
        "left_front_wheel_joint",
        "left_back_wheel_joint",
        "gripper_extension",
        "left_gripper_joint",
        "right_gripper_joint",
        "head_swivel",
    ]
    position = [
        -0.18661060362323356,
        -0.5604601294004192,
        -0.42474332676534,
        -0.49260172808287983,
        -0.028766000000000014,
        0.25180600000000003,
        0.27252040000000005,
        0.11875220230569417,
    ]
    state = collections.namedtuple("JointState", ["name", "position"])(name, position)
    return state


@pytest.fixture(scope="session")
def link_pose_tutorial(urdf_file_tutorial):
    Point = collections.namedtuple("Point", ["x", "y", "z"])
    Quaternion = collections.namedtuple("Quaternion", ["x", "y", "z", "w"])
    Pose = collections.namedtuple("Pose", ["position", "orientation"])
    PoseStamped = collections.namedtuple("PoseStamped", ["pose"])

    poses = {
        "base_link": [0, 0, 0, 0, 0, 0, 1],
        "box": [0.18012, 0.021491, 0.4414, 0, 0, 0.059341, 0.99824],
        "gripper_pole": [0.16123, 0, 0.2, 0, 0, 0, 1],
        "head": [0, 0, 0.3, 0, 0, 0.059341, 0.99824],
        "left_back_wheel": [-0.13333, 0.22, -0.435, 0, -0.24382, 0, 0.96982],
        "left_base": [0, 0.22, -0.35, 0, 0, 0, 1],
        "left_front_wheel": [0.13333, 0.22, -0.435, 0, -0.21078, 0, 0.97753],
        "left_gripper": [0.36123, 0.01, 0.2, 0, 0, 0.12557, 0.99208],
        "left_leg": [0, 0.22, 0.25, 0, 0, 0, 1],
        "left_tip": [0.36123, 0.01, 0.2, 0, 0, 0.12557, 0.99208],
        "right_back_wheel": [-0.13333, -0.22, -0.435, 0, -0.27658, 0, 0.96099],
        "right_base": [0, -0.22, -0.35, 0, 0, 0, 1],
        "right_front_wheel": [0.13333, -0.22, -0.435, 0, -0.09317, 0, 0.99565],
        "right_gripper": [0.36123, -0.01, 0.2, 0, 0, -0.13584, 0.99073],
        "right_leg": [0, -0.22, 0.25, 0, 0, 0, 1],
        "right_tip": [0.36123, -0.01, 0.2, 0, 0, -0.13584, 0.99073],
    }
    poses = {k: [float(e) for e in v] for k, v in poses.items()}
    return {
        k: PoseStamped(
            Pose(
                Point(*v[:3]),
                Quaternion(*v[3:]),
            )
        )
        for k, v in poses.items()
    }
