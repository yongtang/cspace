import pytest

import cspace.transformers

import pathlib
import logging
import torch


def test_kinematics(
    device, urdf_file_tutorial, joint_state_tutorial, link_pose_tutorial
):
    kinematics = cspace.transformers.Kinematics(
        pathlib.Path(urdf_file_tutorial).read_text(), "left_gripper"
    )

    joint_state_tutorial = dict(
        zip(joint_state_tutorial.name, joint_state_tutorial.position)
    )
    state = cspace.torch.classes.JointStateCollection(
        kinematics.joint, tuple(joint_state_tutorial[name] for name in kinematics.joint)
    )

    true_position = torch.tensor(
        [
            link_pose_tutorial["left_gripper"].pose.position.x,
            link_pose_tutorial["left_gripper"].pose.position.y,
            link_pose_tutorial["left_gripper"].pose.position.z,
        ],
        device=device,
        dtype=torch.float64,
    )
    true_orientation = torch.tensor(
        [
            link_pose_tutorial["left_gripper"].pose.orientation.x,
            link_pose_tutorial["left_gripper"].pose.orientation.y,
            link_pose_tutorial["left_gripper"].pose.orientation.z,
            link_pose_tutorial["left_gripper"].pose.orientation.w,
        ],
        device=device,
        dtype=torch.float64,
    )

    pose = kinematics.forward(state)
    assert pose.position("left_gripper").shape == true_position.shape
    assert torch.allclose(pose.position("left_gripper"), true_position, atol=1e-4)
    assert pose.orientation("left_gripper").shape == true_orientation.shape
    assert torch.allclose(pose.orientation("left_gripper"), true_orientation, atol=1e-4)


@pytest.mark.parametrize(
    "model,seed,total,batch,noise,epoch",
    [
        pytest.param(
            "gpt2", 12345, 1 * 1024 * 1024, 32 * 1024, 8, 5, marks=pytest.mark.full
        ),
        pytest.param("gpt2", 12345, 8, 2, 2, 5),
        pytest.param("gpt2", 12345, 8, 2, None, 5),
    ],
)
def test_train(
    device,
    urdf_file_tutorial,
    joint_state_tutorial,
    link_pose_tutorial,
    model,
    seed,
    total,
    batch,
    noise,
    epoch,
):
    kinematics = cspace.transformers.Kinematics(
        pathlib.Path(urdf_file_tutorial).read_text(), "left_gripper", model=model
    )
    kinematics.train(
        seed=seed, total=total, batch=batch, noise=noise, epoch=epoch, device=device
    )

    joint_state_tutorial = dict(
        zip(joint_state_tutorial.name, joint_state_tutorial.position)
    )
    state = cspace.torch.classes.JointStateCollection(
        kinematics.joint, tuple(joint_state_tutorial[name] for name in kinematics.joint)
    )

    pose = kinematics.forward(state)

    inverse = kinematics.inverse(pose)
    logging.getLogger(__name__).info(
        ("\n[Inverse Kinematics]" + "\nPred: {}" + "\nTrue: {}\n").format(
            list(
                (name, inverse.position(kinematics.spec, name)) for name in inverse.name
            ),
            list((name, state.position(kinematics.spec, name)) for name in state.name),
        )
    )
