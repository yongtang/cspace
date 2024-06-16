import cspace.cspace.classes
import cspace.torch.ops
import dataclasses
import functools
import torch


class LinkPoseCollection(cspace.cspace.classes.LinkPoseCollection):
    def __init__(self, base, name, position, orientation):
        self._base_ = base
        self._name_ = tuple(name)
        self._position_ = torch.as_tensor(position, dtype=torch.float64)
        assert len(self._name_) == self._position_.shape[-1]
        self._orientation_ = torch.as_tensor(orientation, dtype=torch.float64)
        assert len(self._name_) == self._orientation_.shape[-1]

        @functools.cache
        def f_index(self, name):
            return self.name.index(name)

        self._f_index_ = functools.partial(f_index, self)

    @property
    def base(self):
        return self._base_

    @property
    def name(self):
        return self._name_

    def position(self, name):
        return torch.select(self._position_, dim=-1, index=self._f_index_(name))

    def orientation(self, name):
        return torch.select(self._orientation_, dim=-1, index=self._f_index_(name))

    def delta(self, spec, other):
        assert self.base == other.base
        assert self.name == other.name
        assert self._position_.shape == other._position_.shape
        assert self._orientation_.shape == other._orientation_.shape

        def f_delta(spec, name, self, other):
            self_transform = Transform(
                xyz=self.position(name),
                rot=cspace.torch.ops.qua_to_rot(self.orientation(name)),
            )
            other_transform = Transform(
                xyz=other.position(name),
                rot=cspace.torch.ops.qua_to_rot(other.orientation(name)),
            )

            delta_transform = self_transform.inverse() * other_transform

            delta_scale = torch.special.expit(
                cspace.torch.ops.se3_log(delta_transform.xyz, delta_transform.rot)
            )
            return delta_scale

        return torch.stack(
            tuple(f_delta(spec, name, self, other) for name in self.name), dim=-1
        )

    @property
    def batch(self):
        return tuple(self._position_.shape[:-2])


class JointStateCollection(cspace.cspace.classes.JointStateCollection):
    def __init__(self, name, position):
        self._name_ = tuple(name)
        self._position_ = torch.as_tensor(position, dtype=torch.float64)
        assert len(self._name_) == self._position_.shape[-1]

        @functools.cache
        def f_index(self, name):
            if name not in self.name:
                return -1
            return self.name.index(name)

        self._f_index_ = functools.partial(f_index, self)

    @property
    def name(self):
        return self._name_

    def position(self, name):
        index = self._f_index_(name)
        if index < 0:  # TODO: mimic
            return torch.empty(
                self._position_.shape[:-1],
                device=self._position_.device,
                dtype=self._position_.dtype,
            )
        return torch.select(self._position_, dim=-1, index=index)

    def identity(self):
        xyz = torch.as_tensor(
            [0, 0, 0], device=self._position_.device, dtype=self._position_.dtype
        )
        rot = rot = torch.eye(
            3, device=self._position_.device, dtype=self._position_.dtype
        )

        return Transform(xyz=xyz, rot=rot)

    def forward(self, spec, *link, base=None):
        base = base if base else spec.base
        link = link if len(link) else spec.link
        entries = tuple(self.transform(spec, item, base) for item in link)
        position = torch.stack(tuple(entry.xyz for entry in entries), dim=-1)
        orientation = torch.stack(tuple(entry.qua for entry in entries), dim=-1)
        return LinkPoseCollection(base, link, position, orientation)

    def apply(self, spec, delta):
        assert self._position_.shape == delta.shape, "{} vs. {}".format(
            self._position_.shape, delta.shape
        )

        def f_apply(joint, value, delta):
            delta_scale = torch.clip(delta, min=0.0, max=1.0)

            self_value = value
            self_value = (
                self.angle(self_value, joint.motion.zero, joint.motion.limit)
                if joint.motion.call == "angular"
                else self_value
            )
            self_value = torch.clip(
                self_value,
                min=joint.motion.zero - joint.motion.limit,
                max=joint.motion.zero + joint.motion.limit,
            )
            self_scale = (self_value - (joint.motion.zero - joint.motion.limit)) / (
                joint.motion.limit * 2.0
            )

            other_scale = (self_scale + delta_scale + 1.0) % 1.0

            other_value = other_scale * (joint.motion.limit * 2.0) + (
                joint.motion.zero - joint.motion.limit
            )

            other_value = torch.clip(
                other_value,
                min=joint.motion.zero - joint.motion.limit,
                max=joint.motion.zero + joint.motion.limit,
            )

            return other_value

        position = torch.stack(
            tuple(
                f_apply(
                    spec.joint(name),
                    self.position(name),
                    torch.select(delta, dim=-1, index=index),
                )
                for index, name in enumerate(self.name)
            ),
            dim=-1,
        )
        return self.__class__(name=self.name, position=position)

    def delta(self, spec, other):
        assert self.name == other.name

        def f_delta(spec, name, self, other):
            joint = spec.joint(name)

            self_value = self.position(name)
            self_value = (
                self.angle(self_value, joint.motion.zero, joint.motion.limit)
                if joint.motion.call == "angular"
                else self_value
            )
            self_value = torch.clip(
                self_value,
                min=joint.motion.zero - joint.motion.limit,
                max=joint.motion.zero + joint.motion.limit,
            )
            self_scale = (self_value - (joint.motion.zero - joint.motion.limit)) / (
                joint.motion.limit * 2.0
            )

            other_value = other.position(name)
            other_value = (
                self.angle(other_value, joint.motion.zero, joint.motion.limit)
                if joint.motion.call == "angular"
                else other_value
            )
            other_value = torch.clip(
                other_value,
                min=joint.motion.zero - joint.motion.limit,
                max=joint.motion.zero + joint.motion.limit,
            )
            other_scale = (other_value - (joint.motion.zero - joint.motion.limit)) / (
                joint.motion.limit * 2.0
            )

            delta_scale = (other_scale - self_scale + 1.0) % 1.0

            delta_scale = torch.clip(delta_scale, min=0.0, max=1.0)

            return delta_scale

        return torch.stack(
            tuple(f_delta(spec, name, self, other) for name in self.name),
            dim=-1,
        )

    @property
    def batch(self):
        return tuple(self._position_.shape[:-1])

    @classmethod
    def zero(cls, spec, joint, batch=None):
        batch = batch if batch else []
        return cls(
            joint,
            torch.stack(
                tuple(
                    torch.tensor(spec.joint(name).motion.zero).expand(batch)
                    for name in joint
                ),
                dim=-1,
            ),
        )

    @classmethod
    def stack(cls, collections):
        name = set(collection.name for collection in collections)
        assert len(name) == 1
        name = next(iter(name))

        position = torch.stack(
            tuple(collection._position_ for collection in collections)
        )

        return cls(name=name, position=position)

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
        assert len(batch) == 1, "{} vs. {}".format(xyz.shape, rot.shape)
        assert xyz.shape[-1:] == torch.Size([3]) and rot.shape[-2:] == torch.Size(
            [3, 3]
        ), "{} vs. {}".format(xyz.shape, rot.shape)
        object.__setattr__(self, "xyz", xyz)
        object.__setattr__(self, "rot", rot)

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
