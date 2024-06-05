import cspace.transformers

import itertools
import pathlib
import numpy
import torch
import math


def test_kinematics(device, urdf_file, joint_state, link_pose):
    kinematics = cspace.transformers.Kinematics(
        pathlib.Path(urdf_file).read_text(), "left_gripper"
    )

    spec = cspace.cspace.classes.Spec(description=pathlib.Path(urdf_file).read_text())

    joint_state = dict(zip(joint_state.name, joint_state.position))
    name = tuple(joint.name for joint in spec.joint if joint.type != "fixed")
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
        if joint.limit.lower < joint.limit.upper:
            value = joint.limit.lower + (joint.limit.upper - joint.limit.lower) * value
        return (joint.name, float(value))

    joints = tuple(joint for joint in kinematics.spec.joint if joint.type != "fixed")
    values = numpy.random.default_rng(12345).random(len(joints))
    entries = tuple(f_joint(joint, value) for joint, value in zip(joints, values))

    name = tuple(name for name, entry in entries)
    position = torch.tensor(
        tuple(entry for name, entry in entries), dtype=torch.float64
    )
    state = cspace.torch.classes.JointStateCollection(name, position)

    encoded = kinematics.tokenize(state)

    entries = tuple(
        cspace.transformers.DataEncoding.ScaleRecord(
            e="joint", name=name, index=0, entry=(state(name).position)
        )
        for name in state.name
    )
    count = math.ceil(max(abs(entry.entry) for entry in entries))
    entries = tuple(
        [
            cspace.transformers.DataEncoding.ScaleRecord(
                e=entry.e,
                name=entry.name,
                index=entry.index,
                entry=(entry.entry / count if count > 0 else 0.0),
            )
        ]
        * count
        for entry in entries
    )
    entries = tuple(
        entry
        for entry in itertools.chain.from_iterable(zip(*entries))
        if abs(entry.entry) > 0.0
    )

    def f_link(spec, link, base, zero):
        zero_transform = zero.transform(spec, link, base)
        link_transform = state.transform(spec, link, base)
        transform = zero_transform.inverse() * link_transform
        return [
            cspace.transformers.DataEncoding.ScaleRecord(
                e="link", name=link, index=index, entry=entry
            )
            for index, entry in enumerate(
                cspace.torch.ops.se3_log(transform.xyz, transform.rot)
            )
        ]

    zero = cspace.torch.classes.JointStateCollection(
        state.name, tuple(0.0 for name in state.name)
    )
    links = tuple(
        itertools.chain.from_iterable(
            tuple(
                f_link(kinematics.spec, link, kinematics.base, zero)
                for link in kinematics.link
            )
        )
    )
    count = math.ceil(max(abs(entry.entry) for entry in links))
    links = tuple(
        [
            cspace.transformers.DataEncoding.ScaleRecord(
                e=entry.e,
                name=entry.name,
                index=entry.index,
                entry=(entry.entry / count if count > 0 else 0.0),
            )
        ]
        * count
        for entry in links
    )
    links = tuple(
        entry
        for entry in itertools.chain.from_iterable(zip(*links))
        if abs(entry.entry) > 0.0
    )

    encoding = cspace.transformers.DataEncoding(kinematics.spec, kinematics.link)
    none = tuple(
        [
            cspace.transformers.DataEncoding.ScaleRecord(
                e=None, name=None, index=None, entry=None
            )
        ]
    )
    true = links + none + entries + none
    true = tuple(map(encoding.encode, true))
    assert len(true) == len(encoded)
    assert all([(a == b) for a, b in zip(true, encoded)])

    decoded = kinematics.assembly(encoded)

    assert state.name == decoded.name
    assert torch.allclose(
        torch.stack(tuple(state(name).position for name in state.name)),
        torch.stack(tuple(decoded(name).position for name in decoded.name)),
        atol=1e-2,
    )
