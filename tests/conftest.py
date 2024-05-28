import pytest

import requests
import collections
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
        "right_base": [0, -0.22, -0.35, 0, 0, 0, 1],
        "right_gripper": [0.36123, -0.01, 0.2, 0, 0, -0.13584, 0.99073],
        "right_leg": [0, -0.22, 0.25, 0, 0, 0, 1],
        "right_tip": [0.36123, -0.01, 0.2, 0, 0, -0.13584, 0.99073],
    }
    poses = dict((k, [float(e) for e in v]) for k, v in poses.items())
    return dict(
        (
            k,
            PoseStamped(
                Pose(
                    Point(*v[:3]),
                    Quaternion(*v[3:]),
                )
            ),
        )
        for k, v in poses.items()
    )
