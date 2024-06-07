import pytest

import cspace.transformers

import itertools
import pathlib
import logging
import numpy
import torch


def test_kinematics(device, urdf_file, joint_state, link_pose):
    kinematics = cspace.transformers.Kinematics(
        pathlib.Path(urdf_file).read_text(), "left_gripper"
    )

    spec = cspace.cspace.classes.Spec(description=pathlib.Path(urdf_file).read_text())

    joint_state = dict(zip(joint_state.name, joint_state.position))
    name = tuple(joint.name for joint in spec.joint if joint.motion.call)
    position = torch.tensor(
        tuple(joint_state[entry] for entry in name), dtype=torch.float64, device=device
    )
    state = cspace.torch.classes.JointStateCollection(name, position)

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

    def f_joint(joint, value):
        return (joint.name, float(value))

    joints = tuple(joint for joint in kinematics.spec.joint if joint.motion.call)
    values = numpy.random.default_rng(12345).random(len(joints))
    entries = tuple(f_joint(joint, value) for joint, value in zip(joints, values))

    name = tuple(name for name, entry in entries)
    position = torch.tensor(
        tuple(entry for name, entry in entries), dtype=torch.float64
    )
    state = cspace.torch.classes.JointStateCollection(name, position)

    encoded = kinematics.tokenize(state)

    def f_pose(spec, link, base, zero, link_transforms):
        zero_transform = zero.transform(spec, link, base)
        link_transform = link_transforms[link]
        transform = zero_transform.inverse() * link_transform
        return [
            kinematics.encoding.encode(
                (cspace.transformers.PoseIndex(name=link, field=field), entry)
            )
            for field, entry in enumerate(
                cspace.torch.ops.se3_log(transform.xyz, transform.rot)
            )
        ]

    link_transforms = {
        link: state.transform(kinematics.spec, link, kinematics.base)
        for link in kinematics.link
    }
    name = tuple(joint.name for joint in kinematics.spec.joint if joint.motion.call)
    zero = cspace.torch.classes.JointStateCollection(name, tuple(0 for e in name))
    entries = tuple(
        itertools.chain.from_iterable(
            tuple(
                f_pose(kinematics.spec, link, kinematics.base, zero, link_transforms)
                for link in kinematics.link
            )
        )
    )
    entries = entries + tuple(
        [kinematics.encoding.encode((cspace.transformers.NoneIndex(), 0))]
    )
    entries = entries + tuple(
        kinematics.encoding.encode(
            (cspace.transformers.JointIndex(name=name), state(name).position)
        )
        for name in state.name
    )
    entries = entries + tuple(
        [kinematics.encoding.encode((cspace.transformers.NoneIndex(), 0))]
    )
    assert entries == encoded

    decoded = kinematics.assembly(encoded)

    assert state.name == decoded.name
    assert torch.allclose(
        torch.stack(tuple(state(name).position for name in state.name)),
        torch.stack(tuple(decoded(name).position for name in decoded.name)),
        atol=1e-3,
    )


@pytest.mark.train
@pytest.mark.parametrize(
    "model,seed,total,epoch,batch",
    [
        ("gpt2", 12345, 1024 * 1024, 20, 2048),
    ],
)
def test_train(
    device, urdf_file, joint_state, link_pose, model, seed, total, batch, epoch
):
    kinematics = cspace.transformers.Kinematics(
        pathlib.Path(urdf_file).read_text(), "left_gripper", model=model
    )
    kinematics.train(seed=seed, total=total, batch=batch, epoch=epoch, device=device)

    def f_joint(joint, value):
        return (joint.name, float(value))

    joints = tuple(joint for joint in kinematics.spec.joint if joint.motion.call)
    values = numpy.random.default_rng(seed).random(len(joints))
    entries = tuple(f_joint(joint, value) for joint, value in zip(joints, values))

    name = tuple(name for name, entry in entries)
    position = torch.tensor(
        tuple(entry for name, entry in entries), dtype=torch.float64
    )
    state = cspace.torch.classes.JointStateCollection(name, position)

    pose = kinematics.forward(state)

    inverse = kinematics.inverse(pose)
    logging.getLogger(__name__).info(
        ("\n[Inverse Kinematics]" + "\nPred: {}" + "\nTrue: {}" + "\n").format(
            inverse._position_, state._position_
        )
    )
