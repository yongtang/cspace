import torch
import numpy


def rpy_to_rot(rpy):
    rpy = torch.as_tensor(rpy)

    r, p, y = torch.unbind(rpy, dim=-1)

    sr, cr = torch.sin(r), torch.cos(r)
    sp, cp = torch.sin(p), torch.cos(p)
    sy, cy = torch.sin(y), torch.cos(y)
    cc, cs = cr * cy, cr * sy
    sc, ss = sr * cy, sr * sy

    rot = torch.stack(
        (
            cp * cy,
            sp * sc - cs,
            sp * cc + ss,
            cp * sy,
            sp * ss + cc,
            sp * cs - sc,
            -sp,
            cp * sr,
            cp * cr,
        ),
        dim=-1,
    )

    return torch.unflatten(rot, -1, (3, 3))


def rot_to_rpy(rot):
    rot = torch.as_tensor(rot)

    cy = torch.sqrt(rot[..., 0, 0] * rot[..., 0, 0] + rot[..., 1, 0] * rot[..., 1, 0])

    ax = torch.where(
        (cy > numpy.finfo(float).eps * 4.0),
        torch.atan2(rot[..., 2, 1], rot[..., 2, 2]),
        torch.atan2(-rot[..., 1, 2], rot[..., 1, 1]),
    )
    ay = torch.where(
        (cy > numpy.finfo(float).eps * 4.0),
        torch.atan2(-rot[..., 2, 0], cy),
        torch.atan2(-rot[..., 2, 0], cy),
    )
    az = torch.where(
        (cy > numpy.finfo(float).eps * 4.0),
        torch.atan2(rot[..., 1, 0], rot[..., 0, 0]),
        torch.zeros_like(ax),
    )

    return torch.stack((ax, ay, az), dim=-1)


def qua_to_rpy(qua):
    return rot_to_rpy(qua_to_rot(qua))


def rpy_to_qua(rpy):
    rpy = torch.as_tensor(rpy)

    r, p, y = torch.unbind(rpy, dim=-1)

    sr, cr = torch.sin(r / 2.0), torch.cos(r / 2.0)
    sp, cp = torch.sin(p / 2.0), torch.cos(p / 2.0)
    sy, cy = torch.sin(y / 2.0), torch.cos(y / 2.0)

    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    qw = cr * cp * cy + sr * sp * sy

    return torch.stack((qx, qy, qz, qw), dim=-1)


def qua_to_rot(qua):
    qua = torch.as_tensor(qua)

    qx, qy, qz, qw = torch.unbind(qua, dim=-1)

    rot = torch.stack(
        (
            qw * qw + qx * qx - qy * qy - qz * qz,
            2.0 * (qx * qy - qw * qz),
            2.0 * (qw * qy + qx * qz),
            2.0 * (qx * qy + qw * qz),
            qw * qw - qx * qx + qy * qy - qz * qz,
            2.0 * (qy * qz - qw * qx),
            2.0 * (qx * qz - qw * qy),
            2.0 * (qw * qx + qy * qz),
            qw * qw - qx * qx - qy * qy + qz * qz,
        ),
        dim=-1,
    )

    return torch.unflatten(rot, -1, (3, 3))


def rot_to_qua(rot):
    rot = torch.as_tensor(rot)

    zero = torch.zeros_like(rot[..., 0, 0])

    K = torch.stack(
        (
            rot[..., 0, 0] - rot[..., 1, 1] - rot[..., 2, 2],
            zero,
            zero,
            zero,
            rot[..., 0, 1] + rot[..., 1, 0],
            rot[..., 1, 1] - rot[..., 0, 0] - rot[..., 2, 2],
            zero,
            zero,
            rot[..., 0, 2] + rot[..., 2, 0],
            rot[..., 1, 2] + rot[..., 2, 1],
            rot[..., 2, 2] - rot[..., 0, 0] - rot[..., 1, 1],
            zero,
            rot[..., 2, 1] - rot[..., 1, 2],
            rot[..., 0, 2] - rot[..., 2, 0],
            rot[..., 1, 0] - rot[..., 0, 1],
            rot[..., 0, 0] + rot[..., 1, 1] + rot[..., 2, 2],
        ),
        dim=-1,
    )

    K = torch.unflatten(K, -1, (4, 4)) / 3.0
    L, Q = torch.linalg.eigh(K)

    _, index = torch.max(L, dim=-1, keepdim=True)
    index = torch.stack((index, index, index, index), dim=-2)

    qua = torch.gather(Q, dim=-1, index=index)
    qua = torch.squeeze(qua, -1)
    qua = torch.where(qua[..., 3:4] < 0.0, -qua, qua)

    return qua
