import pytest

import cspace.transformers

import pathlib
import logging
import torch


def test_kinematics(device, urdf_file, joint_state, link_pose):
    kinematics = cspace.transformers.Kinematics(
        pathlib.Path(urdf_file).read_text(), "left_gripper"
    )

    joint_state = dict(zip(joint_state.name, joint_state.position))
    state = cspace.torch.classes.JointStateCollection(
        kinematics.joint, tuple(joint_state[name] for name in kinematics.joint)
    )

    true_position = torch.tensor(
        [
            link_pose["left_gripper"].pose.position.x,
            link_pose["left_gripper"].pose.position.y,
            link_pose["left_gripper"].pose.position.z,
        ],
        device=device,
        dtype=torch.float64,
    )
    true_orientation = torch.tensor(
        [
            link_pose["left_gripper"].pose.orientation.x,
            link_pose["left_gripper"].pose.orientation.y,
            link_pose["left_gripper"].pose.orientation.z,
            link_pose["left_gripper"].pose.orientation.w,
        ],
        device=device,
        dtype=torch.float64,
    )

    pose = kinematics.forward(state)
    assert pose("left_gripper").position.shape == true_position.shape
    assert torch.allclose(pose("left_gripper").position, true_position, atol=1e-4)
    assert pose("left_gripper").orientation.shape == true_orientation.shape
    assert torch.allclose(pose("left_gripper").orientation, true_orientation, atol=1e-4)


@pytest.mark.train
@pytest.mark.parametrize(
    "model,seed,total,epoch,batch",
    [
        ("gpt2", 12345, 16 * 1024, 20, 2048),
    ],
)
def test_train(
    device, urdf_file, joint_state, link_pose, model, seed, total, batch, epoch
):
    kinematics = cspace.transformers.Kinematics(
        pathlib.Path(urdf_file).read_text(), "left_gripper", model=model
    )
    kinematics.train(seed=seed, total=total, batch=batch, epoch=epoch, device=device)

    joint_state = dict(zip(joint_state.name, joint_state.position))
    state = cspace.torch.classes.JointStateCollection(
        kinematics.joint, tuple(joint_state[name] for name in kinematics.joint)
    )

    pose = kinematics.forward(state)

    inverse = kinematics.inverse(pose)
    logging.getLogger(__name__).info(
        ("\n[Inverse Kinematics]" + "\nPred: {}" + "\nTrue: {}" + "\n").format(
            inverse._position_, state._position_
        )
    )
