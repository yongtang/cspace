import pytest

import cspace.transformers

import accelerate
import pathlib
import torch


def test_kinematics_forward(
    device, urdf_file_tutorial, joint_state_tutorial, link_pose_tutorial
):
    kinematics = cspace.transformers.InverseKinematics(
        pathlib.Path(urdf_file_tutorial).read_text(), "left_gripper"
    )

    joint_state_tutorial = dict(
        zip(joint_state_tutorial.name, joint_state_tutorial.position)
    )
    state = cspace.torch.classes.JointStateCollection(
        kinematics.joint,
        torch.tensor(
            tuple(joint_state_tutorial[name] for name in kinematics.joint),
            device=device,
        ),
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
    "model,bucket,length,noise,total,batch,repeat",
    [
        pytest.param(
            "gpt2",
            2,
            10,
            None,
            8 * 1024 * 1024,
            32 * 1024,
            5,
            marks=pytest.mark.full,
        ),
        pytest.param("gpt2", 2, 10, None, 8, 2, 5),
        pytest.param("gpt2", 2, 10, 2, 8, 2, 5),
    ],
)
def test_kinematics_inverse(
    device,
    urdf_file_tutorial,
    joint_state_tutorial,
    link_pose_tutorial,
    model,
    bucket,
    length,
    noise,
    total,
    batch,
    repeat,
    request,
    tmp_path_factory,
):
    saved = pathlib.Path.joinpath(
        tmp_path_factory.mktemp("model"),
        f"{request.node.name}-{request.node.callspec.id}.pth",
    )

    accelerator = accelerate.Accelerator()
    logger = accelerate.logging.get_logger(__name__, log_level="INFO")

    kinematics = cspace.transformers.InverseKinematics(
        pathlib.Path(urdf_file_tutorial).read_text(),
        "left_gripper",
        model=model,
        bucket=bucket,
        length=length,
    )

    kinematics.train(
        logger=logger,
        accelerator=accelerator,
        save=saved,
        batch=batch,
        total=total,
        repeat=repeat,
        noise=noise,
    )

    kinematics = torch.load(saved)

    joint_state_tutorial = dict(
        zip(joint_state_tutorial.name, joint_state_tutorial.position)
    )
    state = cspace.torch.classes.JointStateCollection(
        kinematics.joint, tuple(joint_state_tutorial[name] for name in kinematics.joint)
    )

    pose = kinematics.forward(state)

    inverse = kinematics.inverse(pose)
    pred = kinematics.forward(inverse)

    zero = cspace.torch.classes.JointStateCollection.apply(
        kinematics.spec,
        kinematics.joint,
        torch.zeros([len(kinematics.joint)]),
        min=-1.0,
        max=1.0,
    )
    mark = kinematics.forward(zero)
    logger.info(
        (
            "\n"
            + "[Inverse Kinematics]\n"
            + "\n"
            + "Limit:{}\n"
            + "\n"
            + "--------------------\n"
            + "\n"
            + "Zero: {}\n"
            + "\n"
            + "Pose: [position]    {}\n"
            + "      [orientation] {}\n"
            + "\n"
            + "--------------------\n"
            + "\n"
            + "True: {}\n"
            + "\n"
            + "Pose: [position]    {}\n"
            + "      [orientation] {}\n"
            + "\n"
            + "--------------------\n"
            + "\n"
            + "Pred: {}\n"
            + "\n"
            + "Pose: [position]    {}\n"
            + "      [orientation] {}\n"
            + "\n"
        ).format(
            list(
                (name, kinematics.spec.joint(name).motion.limit) for name in zero.name
            ),
            list(
                (name, zero.position(kinematics.spec, name).data.cpu().item())
                for name in zero.name
            ),
            list((name, mark.position(name).data.cpu().tolist()) for name in mark.name),
            list(
                (name, mark.orientation(name).data.cpu().tolist()) for name in mark.name
            ),
            list(
                (name, state.position(kinematics.spec, name).data.cpu().item())
                for name in state.name
            ),
            list((name, pose.position(name).data.cpu().tolist()) for name in pose.name),
            list(
                (name, pose.orientation(name).data.cpu().tolist()) for name in pose.name
            ),
            list(
                (name, inverse.position(kinematics.spec, name).data.cpu().item())
                for name in inverse.name
            ),
            list(
                (name, pred.position(name).data.cpu().numpy().tolist())
                for name in pred.name
            ),
            list(
                (name, pred.orientation(name).data.cpu().numpy().tolist())
                for name in pred.name
            ),
        )
    )
