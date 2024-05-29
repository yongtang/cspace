import pytest

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
        "--device", action="store", default="cpu,cuda", help="run tests with CUDA"
    )


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

    def f_euler(angle_r, angle_p, angle_y, interleave):
        return scipy.special.radian(
            (angle_r + interleave, angle_p + interleave, angle_y + interleave), 0, 0
        )

    def f_entries(f_t3d, angle_r, angle_p, angle_y, batch, interleave):
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

    angle_r, angle_p, angle_y, batch, interleave = request.param
    entries = (
        f_entries(f_rpy, angle_r, angle_p, angle_y, batch, interleave),
        f_entries(f_qua, angle_r, angle_p, angle_y, batch, interleave),
        f_entries(f_rot, angle_r, angle_p, angle_y, batch, interleave),
        f_entries(f_rpy_to_rot, angle_r, angle_p, angle_y, batch, interleave),
        f_entries(f_rot_to_rpy, angle_r, angle_p, angle_y, batch, interleave),
        f_entries(f_rpy_to_qua, angle_r, angle_p, angle_y, batch, interleave),
        f_entries(f_qua_to_rot, angle_r, angle_p, angle_y, batch, interleave),
        f_entries(f_rot_to_qua, angle_r, angle_p, angle_y, batch, interleave),
        batch,
    )

    return entries


@pytest.fixture(scope="session")
def urdf_file(tmp_path_factory):
    url = (
        "https://raw.githubusercontent.com/ros/urdf_tutorial/ros2/urdf/07-physics.urdf"
    )
    file = tmp_path_factory.mktemp("data") / "07-physics.urdf"
    file.write_bytes(requests.get(url).content)
    return file


@pytest.fixture(scope="session")
def joint_state(urdf_file):
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
def link_pose(urdf_file):
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
