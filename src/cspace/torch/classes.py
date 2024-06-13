import cspace.cspace.classes
import cspace.torch.ops
import dataclasses
import torch


class LinkPose(cspace.cspace.classes.LinkPose):
    pass


class LinkPoseCollection(cspace.cspace.classes.LinkPoseCollection):
    def __init__(self, base, name, position, orientation):
        self._base_ = base
        self._name_ = tuple(name)
        self._position_ = torch.as_tensor(position, dtype=torch.float64)
        assert len(self._name_) == self._position_.shape[-1]
        self._orientation_ = torch.as_tensor(orientation, dtype=torch.float64)
        assert len(self._name_) == self._orientation_.shape[-1]

    @property
    def base(self):
        return self._base_

    @property
    def position(self):
        return self._position_

    @property
    def orientation(self):
        return self._orientation_

    @property
    def name(self):
        return self._name_

    def __call__(self, name):
        index = self.name.index(name)
        return LinkPose(
            base=self.base,
            name=name,
            position=torch.select(self.position, dim=-1, index=index),
            orientation=torch.select(self.orientation, dim=-1, index=index),
        )


class JointState(cspace.cspace.classes.JointState):
    @classmethod
    def origin(cls, position, xyz, rpy):
        xyz = torch.as_tensor(
            xyz,
            device=position.device,
            dtype=position.dtype,
        )
        rpy = torch.as_tensor(
            rpy,
            device=position.device,
            dtype=position.dtype,
        )
        rot = cspace.torch.ops.rpy_to_rot(rpy)

        return Transform(xyz=xyz, rot=rot)

    @classmethod
    def linear(cls, position, sign, axis):
        position = position if sign > 0 else -position

        zero = torch.zeros_like(position)

        if axis == 0:
            xyz = torch.stack((position, zero, zero), dim=-1)
        elif axis == 1:
            xyz = torch.stack((zero, position, zero), dim=-1)
        else:
            xyz = torch.stack((zero, zero, position), dim=-1)

        rot = torch.eye(
            3,
            device=position.device,
            dtype=position.dtype,
        ).expand(*(xyz.shape[:-1] + tuple([-1, -1])))

        return Transform(xyz=xyz, rot=rot)

    @classmethod
    def angular(cls, position, sign, axis):
        position = position if sign > 0 else -position

        zero = torch.zeros_like(position)
        one = torch.ones_like(position)
        sin = torch.sin(position)
        cos = torch.cos(position)

        if axis == 0:
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
        elif axis == 1:
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
            device=position.device,
            dtype=position.dtype,
        ).expand(*(rot.shape[:-2] + tuple([-1])))

        return Transform(xyz=xyz, rot=rot)

    @classmethod
    def angle(cls, position):
        return (position + torch.pi) % (torch.pi * 2) - torch.pi
        raise NotImplementedError

    @classmethod
    def clip(cls, position, lower, upper):
        return torch.clip(position, min=lower, max=upper)


class JointStateCollection(cspace.cspace.classes.JointStateCollection):
    def __init__(self, name, position):
        self._name_ = tuple(name)
        self._position_ = torch.as_tensor(position, dtype=torch.float64)
        assert len(self._name_) == self._position_.shape[-1]

    @property
    def name(self):
        return self._name_

    @property
    def position(self):
        return self._position_

    def __call__(self, name):
        if name not in self.name:
            return JointState(
                name=name,
                position=torch.empty(
                    self.position.shape[:-1],
                    device=self.position.device,
                    dtype=self.position.dtype,
                ),
            )
        index = self.name.index(name)
        return JointState(
            name=name, position=torch.select(self.position, dim=-1, index=index)
        )

    def identity(self):
        xyz = torch.as_tensor(
            [0, 0, 0], device=self.position.device, dtype=self.position.dtype
        )
        rot = rot = torch.eye(3, device=self.position.device, dtype=self.position.dtype)

        return Transform(xyz=xyz, rot=rot)

    def forward(self, spec, *link, base=None):
        base = base if base else spec.base
        link = link if len(link) else spec.link
        entries = tuple(self.transform(spec, item, base) for item in link)
        position = torch.stack(tuple(entry.xyz for entry in entries), dim=-1)
        orientation = torch.stack(tuple(entry.qua for entry in entries), dim=-1)
        return LinkPoseCollection(base, link, position, orientation)


@dataclasses.dataclass(frozen=True, kw_only=True)
class Transform(cspace.cspace.classes.Transform):
    @property
    def rpy(self):
        return cspace.torch.ops.rot_to_rpy(self.rot)

    @property
    def qua(self):
        return cspace.torch.ops.rot_to_qua(self.rot)

    def inverse(self):
        xyz, rot = cspace.torch.ops.se3_inv(self.xyz, self.rot)
        return Transform(xyz=xyz, rot=rot)

    def __mul__(self, other):
        xyz, rot = cspace.torch.ops.se3_mul(self.xyz, self.rot, other.xyz, other.rot)
        return Transform(xyz=xyz, rot=rot)
