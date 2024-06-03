import cspace.cspace.classes
import cspace.torch.ops
import dataclasses
import torch


@dataclasses.dataclass(frozen=True, kw_only=True)
class Transform(cspace.cspace.classes.Transform):
    @property
    def rpy(self):
        return cspace.torch.ops.rot_to_rpy(self.rot)

    @property
    def qua(self):
        return cspace.torch.ops.rot_to_qua(self.rot)

    def inverse(self):
        assert self.rot.ndim == 2 and other.rot.ndim == 2
        xyz = torch.mm(self.rot.transpose(), self.xyz)
        rot = self.rot.transpose()
        return Transform(xyz=xyz, rot=rot)

    def __mul__(self, other):
        assert self.rot.ndim == 2 and other.rot.ndim == 2
        xyz = torch.mm(self.rot, other.xyz.unsqueeze(-1)).squeeze(-1) + self.xyz
        rot = torch.mm(self.rot, other.rot)
        return Transform(xyz=xyz, rot=rot)


class ForwardOp(cspace.cspace.classes.ForwardOp):
    def select(self, data, index):
        data = torch.as_tensor(data, dtype=torch.float64)
        return torch.select(data, dim=-1, index=index)

    def identity(self, data):
        data = torch.as_tensor(data, dtype=torch.float64)

        xyz = torch.as_tensor([0, 0, 0], device=data.device, dtype=data.dtype)
        rot = rot = torch.eye(3, device=data.device, dtype=data.dtype)

        return Transform(xyz=xyz, rot=rot)

    def origin(self, data, xyz, rpy):
        data = torch.as_tensor(data, dtype=torch.float64)

        xyz = torch.as_tensor(
            xyz,
            device=data.device,
            dtype=data.dtype,
        )
        rpy = torch.as_tensor(
            rpy,
            device=data.device,
            dtype=data.dtype,
        )
        rot = cspace.torch.ops.rpy_to_rot(rpy)

        return Transform(xyz=xyz, rot=rot)

    def linear(self, data, axis, upper, lower):
        assert len(axis) == 3
        axis = [(index, sign) for index, sign in enumerate(axis) if sign]
        assert len(axis) == 1
        index, sign = next(iter(axis))
        assert (sign == 1) or (sign == -1)

        data = torch.as_tensor(data, dtype=torch.float64)
        data = torch.clip(data, min=lower, max=upper)
        data = data if sign > 0 else -data

        zero = torch.zeros_like(data)

        if index == 0:
            xyz = torch.stack((data, zero, zero), dim=-1)
        elif index == 1:
            xyz = torch.stack((zero, data, zero), dim=-1)
        else:
            xyz = torch.stack((zero, zero, data), dim=-1)

        rot = torch.eye(
            3,
            device=data.device,
            dtype=data.dtype,
        ).expand(*(xyz.shape[:-1] + tuple([-1, -1])))

        return Transform(xyz=xyz, rot=rot)

    def angular(self, data, axis, upper, lower):
        assert len(axis) == 3
        axis = [(index, sign) for index, sign in enumerate(axis) if sign]
        assert len(axis) == 1
        index, sign = next(iter(axis))
        assert (sign == 1) or (sign == -1)

        data = torch.as_tensor(data, dtype=torch.float64)
        data = (
            torch.clip(data, min=lower, max=upper)
            if (upper is not None or lower is not None)
            else data
        )
        data = data if sign > 0 else -data

        zero = torch.zeros_like(data)
        one = torch.ones_like(data)
        sin = torch.sin(data)
        cos = torch.cos(data)

        if index == 0:
            rot = torch.stack(
                (
                    one,
                    zero,
                    zero,
                    zero,
                    cos,
                    -sin,
                    zero,
                    sin,
                    cos,
                ),
                dim=-1,
            )
        elif index == 1:
            rot = torch.stack(
                (
                    cos,
                    zero,
                    sin,
                    zero,
                    one,
                    zero,
                    -sin,
                    zero,
                    cos,
                ),
                dim=-1,
            )
        else:
            rot = torch.stack(
                (
                    cos,
                    -sin,
                    zero,
                    sin,
                    cos,
                    zero,
                    zero,
                    zero,
                    one,
                ),
                dim=-1,
            )
        rot = torch.unflatten(rot, -1, (3, 3))

        xyz = torch.as_tensor(
            (0.0, 0.0, 0.0),
            device=data.device,
            dtype=data.dtype,
        ).expand(*(rot.shape[:-2] + tuple([-1])))

        return Transform(xyz=xyz, rot=rot)

    def stack(self, *transform):
        xyz = torch.stack(tuple(e.xyz for e in transform), dim=-2)
        rot = torch.stack(tuple(e.rot for e in transform), dim=-3)

        return Transform(xyz=xyz, rot=rot)
        return torch.stack(
            tuple(
                map(
                    lambda entry: torch.concatenate((entry.xyz, entry.qua), dim=-1),
                    transform,
                )
            ),
            dim=-2,
        )


class Spec:
    def __new__(cls, description):
        return cspace.cspace.classes.Spec(description=description, op=ForwardOp())
