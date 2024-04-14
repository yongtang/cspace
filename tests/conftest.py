import pytest

import requests
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
