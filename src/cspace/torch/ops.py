import torch


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
    eps = torch.finfo(rot.dtype).eps

    cy = torch.sqrt(rot[..., 0, 0] * rot[..., 0, 0] + rot[..., 1, 0] * rot[..., 1, 0])

    ax = torch.where(
        (cy > eps * 4.0),
        torch.atan2(rot[..., 2, 1], rot[..., 2, 2]),
        torch.atan2(-rot[..., 1, 2], rot[..., 1, 1]),
    )
    ay = torch.where(
        (cy > eps * 4.0),
        torch.atan2(-rot[..., 2, 0], cy),
        torch.atan2(-rot[..., 2, 0], cy),
    )
    az = torch.where(
        (cy > eps * 4.0),
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


def so3_log(rot):
    rot = torch.as_tensor(rot)
    eps = torch.finfo(rot.dtype).eps

    trace = torch.diagonal(rot, offset=0, dim1=-1, dim2=-2).sum(-1)

    angle = torch.arccos((trace - 1.0) / 2.0)

    angle = torch.unsqueeze(angle, dim=-1)

    rx = rot[..., 2, 1] - rot[..., 1, 2]
    ry = rot[..., 0, 2] - rot[..., 2, 0]
    rz = rot[..., 1, 0] - rot[..., 0, 1]

    axis = torch.stack((rx, ry, rz), dim=-1)

    axis = axis / (torch.sin(angle) * 2.0)
    axa = axis * angle

    c0 = torch.abs(angle) > eps
    axa = torch.where(c0, axa, torch.zeros_like(axa))

    c1 = torch.abs(torch.abs(angle) - torch.pi) > eps
    axa = torch.where(
        c1,
        axa,
        torch.stack(
            (
                torch.sqrt((rot[..., 0, 0] + 1.0) / 2.0) * torch.pi,
                torch.sqrt((rot[..., 1, 1] + 1.0) / 2.0) * torch.pi,
                torch.sqrt((rot[..., 2, 2] + 1.0) / 2.0) * torch.pi,
            ),
            dim=-1,
        ),
    )

    return axa


def so3_exp(axa):
    axa = torch.as_tensor(axa)
    eps = torch.finfo(axa.dtype).eps

    axax, axay, axaz = torch.unbind(axa, dim=-1)
    zero = torch.zeros_like(axax)

    omega = torch.stack(
        (
            zero,
            -axaz,
            axay,
            axaz,
            zero,
            -axax,
            -axay,
            axax,
            zero,
        ),
        dim=-1,
    )
    omega = torch.unflatten(omega, -1, (3, 3))

    theta = torch.linalg.norm(axa, dim=-1)
    theta = torch.unsqueeze(torch.unsqueeze(theta, -1), -1)

    shape = omega.shape
    bmm = torch.reshape(
        torch.bmm(torch.reshape(omega, (-1, 3, 3)), torch.reshape(omega, (-1, 3, 3))),
        shape,
    )

    eye = torch.eye(3).expand(shape[:-2] + (3, 3))
    rot = (
        eye
        + torch.sin(theta) * omega / theta
        + (1.0 - torch.cos(theta)) * bmm / (theta * theta)
    )
    return torch.where(torch.abs(theta) > eps, rot, eye)


def se3_log(xyz, rot):
    xyz = torch.as_tensor(xyz)
    rot = torch.as_tensor(rot, dtype=xyz.dtype)
    batch = {xyz.shape[:-1], rot.shape[:-2]}
    assert len(batch) == 1
    batch = list(next(iter(batch)))

    eps = torch.finfo(rot.dtype).eps

    axa = so3_log(rot)

    axax, axay, axaz = torch.unbind(axa, dim=-1)
    zero = torch.zeros_like(axax)

    omega = torch.stack(
        (
            zero,
            -axaz,
            axay,
            axaz,
            zero,
            -axax,
            -axay,
            axax,
            zero,
        ),
        dim=-1,
    )
    omega = torch.unflatten(omega, -1, (3, 3))

    theta = torch.linalg.norm(axa, dim=-1)
    theta = torch.unsqueeze(theta, -1)

    shape = omega.shape
    bmm = torch.reshape(
        torch.bmm(torch.reshape(omega, (-1, 3, 3)), torch.reshape(omega, (-1, 3, 3))),
        shape,
    )
    value = (
        1.0 - (theta * torch.cos(theta / 2.0)) / (2.0 * torch.sin(theta / 2.0))
    ) / (theta * theta)
    value = torch.unsqueeze(value, -1)
    inv = (
        torch.eye(3, device=value.device, dtype=value.dtype) - omega / 2.0 + value * bmm
    )
    shape = xyz.shape
    bmm = torch.reshape(
        torch.bmm(torch.reshape(inv, (-1, 3, 3)), torch.reshape(xyz, (-1, 3, 1))), shape
    )
    v = torch.where(torch.abs(theta) > eps, bmm, xyz)
    return torch.concatenate((v, axa), dim=-1)


def se3_mul(xyz_a, rot_a, xyz_b, rot_b):
    xyz_a = torch.as_tensor(xyz_a)
    rot_a = torch.as_tensor(rot_a, dtype=xyz_a.dtype)
    xyz_b = torch.as_tensor(xyz_b)
    rot_b = torch.as_tensor(rot_b, dtype=xyz_b.dtype)

    batch_a = {xyz_a.shape[:-1], rot_a.shape[:-2]}
    batch_b = {xyz_b.shape[:-1], rot_b.shape[:-2]}
    assert len(batch_a) == 1
    assert len(batch_b) == 1
    batch_a = list(next(iter(batch_a)))
    batch_b = list(next(iter(batch_b)))
    batch = batch_a if len(batch_a) > len(batch_b) else batch_b
    if len(batch_a) < len(batch):
        xyz_a = xyz_a.expand(batch + [3])
        rot_a = rot_a.expand(batch + [3, 3])
    if len(batch_b) < len(batch):
        xyz_b = xyz_b.expand(batch + [3])
        rot_b = rot_b.expand(batch + [3, 3])

    batch = {xyz_a.shape[:-1], rot_a.shape[:-2], xyz_b.shape[:-1], rot_b.shape[:-2]}
    assert len(batch) == 1
    batch = list(next(iter(batch)))

    xyz_a = torch.reshape(xyz_a, [-1, 3])
    rot_a = torch.reshape(rot_a, [-1, 3, 3])
    xyz_b = torch.reshape(xyz_b, [-1, 3])
    rot_b = torch.reshape(rot_b, [-1, 3, 3])

    rot_a = rot_a.to(rot_b.device)
    xyz_a = xyz_a.to(xyz_b.device)
    xyz = torch.bmm(rot_a, xyz_b.unsqueeze(-1)).squeeze(-1) + xyz_a
    rot = torch.bmm(rot_a, rot_b)

    xyz = torch.reshape(xyz, batch + [3])
    rot = torch.reshape(rot, batch + [3, 3])

    return xyz, rot


def se3_inv(xyz, rot):
    xyz = torch.as_tensor(xyz)
    rot = torch.as_tensor(rot, dtype=xyz.dtype)
    batch = {xyz.shape[:-1], rot.shape[:-2]}
    assert len(batch) == 1
    batch = list(next(iter(batch)))

    xyz = torch.reshape(xyz, [-1, 3])
    rot = torch.reshape(rot, [-1, 3, 3])

    rot = torch.transpose(rot, -2, -1)
    xyz = -torch.bmm(rot, xyz.unsqueeze(-1)).squeeze(-1)

    xyz = torch.reshape(xyz, batch + [3])
    rot = torch.reshape(rot, batch + [3, 3])

    return xyz, rot
