import cspace.cspace.classes
import cspace.torch.ops
import dataclasses
import functools
import torch


class LinkPoseCollection(cspace.cspace.classes.LinkPoseCollection):
    def __init__(self, base, name, position, orientation):
        super().__init__(base=base, name=name)
        self._position_ = torch.as_tensor(position, dtype=torch.float64)
        assert len(self.name) == self._position_.shape[-1]
        self._orientation_ = torch.as_tensor(orientation, dtype=torch.float64)
        assert len(self.name) == self._orientation_.shape[-1]

    def position(self, name):
        return torch.select(self._position_, dim=-1, index=self.index(name))

    def orientation(self, name):
        return torch.select(self._orientation_, dim=-1, index=self.index(name))

    def transform(self, name):
        return Transform(
            xyz=self.position(name),
            rot=cspace.torch.ops.qua_to_rot(self.orientation(name)),
        )

    @property
    def batch(self):
        return tuple(self._position_.shape[:-2])

    @functools.cache
    def index(self, name):
        return self.name.index(name)


class JointStateCollection(cspace.cspace.classes.JointStateCollection):
    def __init__(self, name, position):
        super().__init__(name=name)
        self._position_ = torch.as_tensor(position, dtype=torch.float64)
        assert len(self.name) == self._position_.shape[-1]

    def position(self, spec, name):
        if not spec.joint(name).motion.call:  # fixed
            return torch.empty(
                self._position_.shape[:-1],
                device=self._position_.device,
                dtype=self._position_.dtype,
            )
        mimic = spec.mimic.get(name, None)
        index = self.index(mimic.joint if mimic else name)
        value = torch.select(self._position_, dim=-1, index=index)
        return (value * mimic.multiplier + mimic.offset) if mimic else value

    def forward(self, spec, *link, base=None):
        base = base if base else spec.base
        link = link if len(link) else spec.link
        entries = tuple(self.transform(spec, item, base) for item in link)
        position = torch.stack(tuple(entry.xyz for entry in entries), dim=-1)
        orientation = torch.stack(tuple(entry.qua for entry in entries), dim=-1)
        return LinkPoseCollection(base, link, position, orientation)

    @property
    def batch(self):
        return tuple(self._position_.shape[:-1])

    def scale(self, spec, min, max):
        assert min < max

        scale = torch.stack(
            tuple(
                (
                    (self.position(spec, name) - spec.joint(name).motion.zero)
                    / (spec.joint(name).motion.limit)
                )
                for name in self.name
            ),
            dim=-1,
        )

        zero = (min + max) / 2.0
        limit = (max - min) / 2.0
        scale = zero + scale * limit

        scale = torch.clip(scale, min=min, max=max)

        return scale

    @classmethod
    def apply(cls, spec, joint, scale, min, max):
        assert min < max
        assert scale.shape[-1] == len(joint)

        scale = torch.clip(scale, min=min, max=max)

        zero = (min + max) / 2.0
        limit = (max - min) / 2.0
        scale = (scale - zero) / limit

        position = torch.stack(
            tuple(
                (
                    torch.select(scale, dim=-1, index=index)
                    * (spec.joint(name).motion.limit)
                    + spec.joint(name).motion.zero
                )
                for index, name in enumerate(joint)
            ),
            dim=-1,
        )

        return cls(joint, position)

    @functools.cache
    def index(self, name):
        return self.name.index(name)

    @classmethod
    def identity(cls):
        xyz = torch.zeros(3, dtype=torch.float64)
        rot = torch.eye(3, device=xyz.device, dtype=xyz.dtype)
        return Transform(xyz=xyz, rot=rot)

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


@dataclasses.dataclass(frozen=True, kw_only=True)
class Transform(cspace.cspace.classes.Transform):
    def __post_init__(self):
        xyz = torch.as_tensor(self.xyz)
        rot = torch.as_tensor(self.rot, dtype=xyz.dtype)
        batch = {xyz.shape[:-1], rot.shape[:-2]}
        assert len(batch) == 1, f"{xyz.shape} vs. {rot.shape}"
        assert xyz.shape[-1:] == torch.Size([3]) and rot.shape[-2:] == torch.Size(
            [3, 3]
        ), f"{xyz.shape} vs. {rot.shape}"
        object.__setattr__(self, "xyz", xyz)
        object.__setattr__(self, "rot", rot)

    @property
    def rpy(self):
        return cspace.torch.ops.rot_to_rpy(self.rot)

    @property
    def qua(self):
        return cspace.torch.ops.rot_to_qua(self.rot)

    @property
    def log(self):
        return cspace.torch.ops.se3_log(self.xyz, self.rot)

    def inverse(self):
        xyz, rot = cspace.torch.ops.se3_inv(self.xyz, self.rot)
        return Transform(xyz=xyz, rot=rot)

    def __mul__(self, other):
        xyz, rot = cspace.torch.ops.se3_mul(self.xyz, self.rot, other.xyz, other.rot)
        return Transform(xyz=xyz, rot=rot)


class Kinematics(cspace.cspace.classes.Kinematics):
    pass
