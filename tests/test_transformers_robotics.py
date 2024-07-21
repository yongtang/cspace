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
    "model,seed,total,batch,basis,noise,epoch",
    [
        pytest.param(
            "gpt2", 12345, 8 * 1024 * 1024, 32 * 1024, 3, 2, 5, marks=pytest.mark.full
        ),
        pytest.param("gpt2", 12345, 8, 2, 3, 2, 5),
        pytest.param("gpt2", 12345, 8, 2, None, None, 5),
    ],
)
def test_kinematics_inverse(
    device,
    urdf_file_tutorial,
    joint_state_tutorial,
    link_pose_tutorial,
    model,
    seed,
    total,
    batch,
    basis,
    noise,
    epoch,
    request,
    tmp_path_factory,
):
    saved = pathlib.Path.joinpath(
        tmp_path_factory.mktemp("model"),
        f"{request.node.name}-{request.node.callspec.id}.pth",
    )

    accelerator = accelerate.Accelerator()
    logger = accelerate.logging.get_logger(__name__)

    kinematics = cspace.transformers.InverseKinematics(
        pathlib.Path(urdf_file_tutorial).read_text(),
        "left_gripper",
        basis=basis,
        model=model,
    )

    with accelerator.main_process_first():
        dataset = cspace.transformers.InverseDataset(
            total, kinematics.joint, kinematics.link, basis, noise=noise, seed=seed
        )

    kinematics.train(
        logger=logger,
        accelerator=accelerator,
        dataset=dataset,
        batch=batch,
        epoch=epoch,
        save=saved,
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

    zero = cspace.torch.classes.JointStateCollection.zero(
        kinematics.spec, kinematics.joint
    )
    zerop = kinematics.forward(zero)
    logger.info(
        (
            "\n"
            + "[Inverse Kinematics]\n"
            + "\n"
            + "Zeroe: {}\n"
            + "Pose: [position]    {}\n"
            + "      [orientation] {}\n"
            + "\n"
            + "True: {}\n"
            + "Pose: [position]    {}\n"
            + "      [orientation] {}\n"
            + "\n"
            + "Pred: {}\n"
            + "Pred: [position]    {}\n"
            + "      [orientation] {}\n"
            + "\n"
        ).format(
            list((name, zero.position(kinematics.spec, name)) for name in zero.name),
            list((name, zerop.position(name)) for name in zerop.name),
            list((name, zerop.orientation(name)) for name in zerop.name),
            list((name, state.position(kinematics.spec, name)) for name in state.name),
            list((name, pose.position(name)) for name in pose.name),
            list((name, pose.orientation(name)) for name in pose.name),
            list(
                (name, inverse.position(kinematics.spec, name)) for name in inverse.name
            ),
            list((name, pred.position(name)) for name in pred.name),
            list((name, pred.orientation(name)) for name in pred.name),
        )
    )
