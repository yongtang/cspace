import torch


def qua_to_rpy(qua):
    qx, qy, qz, qw = torch.unbind(qua, dim=-1)

    t0 = +2.0 * (qw * qx + qy * qz)
    t1 = +1.0 - 2.0 * (qx * qx + qy * qy)
    r = torch.atan2(t0, t1)

    t2 = +2.0 * (qw * qy - qz * qx)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    p = torch.asin(t2)

    t3 = +2.0 * (qw * qz + qx * qy)
    t4 = +1.0 - 2.0 * (qy * qy + qz * qz)
    y = torch.atan2(t3, t4)

    return torch.stack((r, p, y), dim=-1)


def rpy_to_qua(rpy):
    r, p, y = torch.unbind(rpy, dim=-1)

    sin_r_2, cos_r_2 = torch.sin(r / 2.0), torch.cos(r / 2.0)
    sin_p_2, cos_p_2 = torch.sin(p / 2.0), torch.cos(p / 2.0)
    sin_y_2, cos_y_2 = torch.sin(y / 2.0), torch.cos(y / 2.0)

    qx = sin_r_2 * cos_p_2 * cos_y_2 - cos_r_2 * sin_p_2 * sin_y_2
    qy = cos_r_2 * sin_p_2 * cos_y_2 + sin_r_2 * cos_p_2 * sin_y_2
    qz = cos_r_2 * cos_p_2 * sin_y_2 - sin_r_2 * sin_p_2 * cos_y_2
    qw = cos_r_2 * cos_p_2 * cos_y_2 + sin_r_2 * sin_p_2 * sin_y_2

    return torch.stack((qx, qy, qz, qw), dim=-1)


def qua_to_rot(qua):
    qx, qy, qz, qw = torch.unbind(qua, dim=-1)

    r00 = 2 * (qw * qw + qx * qx) - 1
    r01 = 2 * (qx * qy - qw * qz)
    r02 = 2 * (qx * qz + qw * qy)
    r0 = torch.stack((r00, r01, r02), dim=-1)

    r10 = 2 * (qx * qy + qw * qz)
    r11 = 2 * (qw * qw + qy * qy) - 1
    r12 = 2 * (qy * qz - qw * qx)
    r1 = torch.stack((r10, r11, r12), dim=-1)

    r20 = 2 * (qx * qz - qw * qy)
    r21 = 2 * (qy * qz + qw * qx)
    r22 = 2 * (qw * qw + qz * qz) - 1
    r2 = torch.stack((r20, r21, r22), dim=-1)

    return torch.stack((r0, r1, r2), dim=-2)


def rot_to_qua(rot):
    m = rot
    # q0 = qw
    t = torch.trace(m)
    q = torch.tensor([0.0, 0.0, 0.0, 0.0])

    if t > 0:
        t = torch.sqrt(t + 1)
        q[3] = 0.5 * t
        t = 0.5 / t
        q[0] = (m[2, 1] - m[1, 2]) * t
        q[1] = (m[0, 2] - m[2, 0]) * t
        q[2] = (m[1, 0] - m[0, 1]) * t

    else:
        i = 0
        if m[1, 1] > m[0, 0]:
            i = 1
        if m[2, 2] > m[i, i]:
            i = 2
        j = (i + 1) % 3
        k = (j + 1) % 3

        t = torch.sqrt(m[i, i] - m[j, j] - m[k, k] + 1)
        q[i] = 0.5 * t
        t = 0.5 / t
        q[3] = (m[k, j] - m[j, k]) * t
        q[j] = (m[j, i] + m[i, j]) * t
        q[k] = (m[k, i] + m[i, k]) * t

    return q
