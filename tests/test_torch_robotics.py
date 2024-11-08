import pytest

import cspace.torch.ops
import cspace.torch.classes
import cspace.cspace.classes

import itertools
import logging
import pathlib
import scipy
import torch


@pytest.mark.parametrize(
    "transforms3d_data",
    list(
        itertools.product(
            range(-180, 180, 30),  # angle_r
            range(-180, 180, 30),  # angle_p
            range(-180, 180, 30),  # angle_r
            [[1, 2, 3], [3, 4, 5]],  # linear
            [[], [1], [2], [1, 1], [2, 2]],  # batch
            [15],  # interleave
        ),
    ),
    indirect=True,
    ids=lambda param: "angle({},{},{})-linear({})-batch({})-interleave({})".format(
        *param
    ),
)
def test_ops(transforms3d_data, device):
    (
        rpy,
        qua,
        rot,
        rpy_to_rot,
        rot_to_rpy,
        rpy_to_qua,
        qua_to_rot,
        rot_to_qua,
        so3_log,
        se3_log,
        se3_mul,
        se3_inv,
        se3_xyz,
        batch,
    ) = transforms3d_data

    (
        rpy,
        qua,
        rot,
        rpy_to_rot,
        rot_to_rpy,
        rpy_to_qua,
        qua_to_rot,
        rot_to_qua,
        so3_log,
        se3_log,
        se3_mul,
        se3_inv,
        se3_xyz,
    ) = (
        torch.as_tensor(rpy, device=device),
        torch.as_tensor(qua, device=device),
        torch.as_tensor(rot, device=device),
        torch.as_tensor(rpy_to_rot, device=device),
        torch.as_tensor(rot_to_rpy, device=device),
        torch.as_tensor(rpy_to_qua, device=device),
        torch.as_tensor(qua_to_rot, device=device),
        torch.as_tensor(rot_to_qua, device=device),
        torch.as_tensor(so3_log, device=device),
        torch.as_tensor(se3_log, device=device),
        torch.as_tensor(se3_mul, device=device),
        torch.as_tensor(se3_inv, device=device),
        torch.as_tensor(se3_xyz, device=device),
    )

    assert rpy.shape == tuple(batch + [3])
    assert qua.shape == tuple(batch + [4])
    assert rot.shape == tuple(batch + [3, 3])
    assert rpy_to_rot.shape == tuple(batch + [3, 3])
    assert rot_to_rpy.shape == tuple(batch + [3])
    assert qua_to_rot.shape == tuple(batch + [3, 3])
    assert rot_to_qua.shape == tuple(batch + [4])
    assert so3_log.shape == tuple(batch + [3])
    assert se3_log.shape == tuple(batch + [6])
    assert se3_mul.shape == tuple(batch + [3, 4])
    assert se3_inv.shape == tuple(batch + [3, 4])
    assert se3_xyz.shape == tuple(batch + [3])

    val = cspace.torch.ops.rpy_to_rot(rpy)
    assert val.shape == rpy_to_rot.shape
    assert torch.allclose(val, rpy_to_rot, atol=1e-4)

    val = cspace.torch.ops.rot_to_rpy(rot)
    assert val.shape == rot_to_rpy.shape
    assert torch.allclose(val, rot_to_rpy, atol=1e-4)

    val = cspace.torch.ops.rpy_to_qua(rpy)
    assert val.shape == rpy_to_qua.shape
    assert torch.allclose(val, rpy_to_qua, atol=1e-4)

    val = cspace.torch.ops.qua_to_rot(qua)
    assert val.shape == qua_to_rot.shape
    assert torch.allclose(val, qua_to_rot, atol=1e-4)

    val = cspace.torch.ops.rot_to_qua(rot)
    assert val.shape == rot_to_qua.shape
    if False:  # device != torch.device("cuda"):  # TODO
        assert torch.allclose(val, rot_to_qua, atol=1e-4)

    val = cspace.torch.ops.so3_log(rot)
    assert val.shape == so3_log.shape
    for index, (v, r) in enumerate(
        zip(val.unsqueeze(0).flatten(0, -2), so3_log.unsqueeze(0).flatten(0, -2))
    ):
        if torch.allclose(
            torch.linalg.norm(v),
            torch.tensor(torch.pi, dtype=rot.dtype, device=device),
        ):
            assert torch.allclose(
                torch.abs(v), torch.abs(r), atol=1e-4
            ), f"{index}: {v} vs. {r}"
        else:
            assert torch.allclose(v, r, atol=1e-4), f"{index}: {v} vs. {r}"

    axa = val
    val = cspace.torch.ops.so3_exp(axa)
    assert val.shape == rot.shape
    for index, (v, r, a) in enumerate(
        zip(
            val.unsqueeze(0).flatten(0, -3),
            rot.unsqueeze(0).flatten(0, -3),
            axa.unsqueeze(0).flatten(0, -2),
        )
    ):
        if torch.allclose(
            torch.linalg.norm(a),
            torch.tensor(torch.pi, dtype=rot.dtype, device=device),
        ):
            assert torch.allclose(
                torch.abs(v), torch.abs(r), atol=1e-4
            ), f"{index}: {v} vs. {r}, - {a}"
        else:
            assert torch.allclose(v, r, atol=1e-4), f"{index}: {v} vs. {r} - {a}"

    val = cspace.torch.ops.se3_log(se3_xyz, rot)
    assert val.shape == se3_log.shape
    for index, (v, l, r) in enumerate(
        zip(
            val.unsqueeze(0).flatten(0, -2),
            se3_log.unsqueeze(0).flatten(0, -2),
            rot.unsqueeze(0).flatten(0, -3),
        )
    ):
        angle = torch.abs(torch.arccos((torch.trace(r) - 1.0) / 2.0))
        if (
            torch.abs(angle - torch.pi) <= torch.finfo(r.dtype).eps
        ):  # skip test on +-180
            logging.getLogger(__name__).info(f"skip {index}: {v}")
            v = torch.tensor(
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=v.dtype, device=device
            )
        assert torch.allclose(v, l, atol=1e-4), f"{index}: {v} vs. {l}"

    val_xyz, val_rot = cspace.torch.ops.se3_exp(val)
    assert val_xyz.shape == se3_xyz.shape
    assert val_rot.shape == rot.shape
    for index, (v, s, l, r) in enumerate(
        zip(
            val_xyz.unsqueeze(0).flatten(0, -2),
            se3_xyz.unsqueeze(0).flatten(0, -2),
            val_rot.unsqueeze(0).flatten(0, -3),
            rot.unsqueeze(0).flatten(0, -3),
        )
    ):
        angle = torch.abs(torch.arccos((torch.trace(r) - 1.0) / 2.0))
        if (
            torch.abs(angle - torch.pi) <= torch.finfo(r.dtype).eps
        ):  # skip test on +-180
            logging.getLogger(__name__).info(f"skip {index}: {v}")
        else:
            assert torch.allclose(v, s, atol=1e-4), f"{index}: {v} vs. {s}"
            assert torch.allclose(l, r, atol=1e-4), f"{index}: {l} vs. {r}"

    val_xyz, val_rot = cspace.torch.ops.se3_mul(se3_xyz, rot, se3_xyz, rot)
    val = torch.concatenate((val_rot, val_xyz.unsqueeze(-1)), dim=-1)
    assert val.shape == se3_mul.shape
    assert torch.allclose(val, se3_mul, atol=1e-4)

    val_xyz, val_rot = cspace.torch.ops.se3_inv(se3_xyz, rot)
    val = torch.concatenate((val_rot, val_xyz.unsqueeze(-1)), dim=-1)
    assert val.shape == se3_inv.shape
    assert torch.allclose(val, se3_inv, atol=1e-4)


def test_spec(device, urdf_file_tutorial):
    spec = cspace.cspace.classes.Spec(
        description=pathlib.Path(urdf_file_tutorial).read_text()
    )

    state = cspace.torch.classes.JointStateCollection(
        spec=spec,
        name=(
            "base_to_right_leg",
            "right_front_wheel_joint",
            "left_gripper_joint",
            "gripper_extension",
        ),
        position=torch.tensor(
            [1.0, 1.0, 1.0, -1.0], device=device, dtype=torch.float64
        ),
    )

    joint = spec.joint("base_to_right_leg")

    xyz = torch.tensor(joint.origin.xyz, device=device, dtype=torch.float64)
    rpy = torch.tensor(joint.origin.rpy, device=device, dtype=torch.float64)
    qua = torch.tensor(
        scipy.spatial.transform.Rotation.from_euler("xyz", rpy.cpu()).as_quat(),
        device=device,
        dtype=torch.float64,
    )

    transform = state.transform(spec, joint.child, joint.parent)

    assert torch.allclose(transform.xyz, xyz, atol=1e-4)
    assert torch.allclose(transform.rpy, rpy, atol=1e-4)
    assert torch.allclose(transform.qua, qua, atol=1e-4)

    # continuous axis=(0, 1, 0)
    joint = spec.joint("right_front_wheel_joint")

    xyz = torch.tensor(joint.origin.xyz, device=device, dtype=torch.float64)
    rpy = torch.tensor((0, 1, 0), device=device, dtype=torch.float64)
    qua = torch.tensor(
        scipy.spatial.transform.Rotation.from_euler("xyz", rpy.cpu()).as_quat(),
        device=device,
        dtype=torch.float64,
    )

    transform = state.transform(spec, joint.child, joint.parent)

    assert torch.allclose(transform.xyz, xyz, atol=1e-4)
    assert torch.allclose(transform.rpy, rpy, atol=1e-4)
    assert torch.allclose(transform.qua, qua, atol=1e-4)

    # revolute axis=(0, 0, 1)
    joint = spec.joint("left_gripper_joint")

    xyz = torch.tensor(joint.origin.xyz, device=device, dtype=torch.float64)
    rpy = torch.tensor(
        (0, 0, 0.548),  # limit=0.548
        device=device,
        dtype=torch.float64,
    )
    qua = torch.tensor(
        scipy.spatial.transform.Rotation.from_euler("xyz", rpy.cpu()).as_quat(),
        device=device,
        dtype=torch.float64,
    )

    transform = state.transform(spec, joint.child, joint.parent)

    assert torch.allclose(transform.xyz, xyz, atol=1e-4)
    assert torch.allclose(transform.rpy, rpy, atol=1e-4)
    assert torch.allclose(transform.qua, qua, atol=1e-4)

    # prismatic axis=(1, 0, 0)
    joint = spec.joint("gripper_extension")

    xyz = torch.tensor(joint.origin.xyz, device=device) + torch.tensor(
        (-0.38, 0, 0),  # limit=-0.38
        device=device,
        dtype=torch.float64,
    )
    rpy = torch.tensor(joint.origin.rpy, device=device, dtype=torch.float64)
    qua = torch.tensor(
        scipy.spatial.transform.Rotation.from_euler("xyz", rpy.cpu()).as_quat(),
        device=device,
        dtype=torch.float64,
    )

    transform = state.transform(spec, joint.child, joint.parent)

    assert torch.allclose(transform.xyz, xyz, atol=1e-4)
    assert torch.allclose(transform.rpy, rpy, atol=1e-4)
    assert torch.allclose(transform.qua, qua, atol=1e-4)


def test_transform(
    device, urdf_file_tutorial, joint_state_tutorial, link_pose_tutorial
):
    spec = cspace.cspace.classes.Spec(
        description=pathlib.Path(urdf_file_tutorial).read_text()
    )

    joint_state_tutorial = dict(
        zip(joint_state_tutorial.name, joint_state_tutorial.position)
    )
    name = tuple(joint.name for joint in spec.joint if joint.motion.call)
    position = torch.tensor(
        tuple(joint_state_tutorial[entry] for entry in name),
        dtype=torch.float64,
        device=device,
    )
    state = cspace.torch.classes.JointStateCollection(spec, name, position)

    for link in spec.link:
        if link in link_pose_tutorial:
            pose = spec.forward(state, link)
            true_position = torch.tensor(
                [
                    link_pose_tutorial[link].pose.position.x,
                    link_pose_tutorial[link].pose.position.y,
                    link_pose_tutorial[link].pose.position.z,
                ],
                device=device,
                dtype=torch.float64,
            )
            true_orientation = torch.tensor(
                [
                    link_pose_tutorial[link].pose.orientation.x,
                    link_pose_tutorial[link].pose.orientation.y,
                    link_pose_tutorial[link].pose.orientation.z,
                    link_pose_tutorial[link].pose.orientation.w,
                ],
                device=device,
                dtype=torch.float64,
            )
            pose_position = pose.position(link).to(device)
            pose_orientation = pose.orientation(link).to(device)
            assert true_position.shape == pose_position.shape
            assert torch.allclose(true_position, pose_position, atol=1e-4)
            assert true_orientation.shape == pose_orientation.shape
            assert torch.allclose(true_orientation, pose_orientation, atol=1e-4)
