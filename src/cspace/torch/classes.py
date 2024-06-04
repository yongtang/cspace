import cspace.cspace.classes
import cspace.torch.ops
import dataclasses
import torch


class LinkPose(cspace.cspace.classes.LinkPose):
    def __init__(self, position, orientation):
        self._position_ = position
        self._orientation_ = orientation

    @property
    def position(self):
        return self._position_

    @property
    def orientation(self):
        return self._orientation_


class LinkPoseCollection(cspace.cspace.classes.LinkPoseCollection):
    def __init__(self, name, transform):
        self._name_ = name
        self._transform_ = transform

    @property
    def name(self):
        return self._name_

    def __call__(self, name):
        transform = self._transform_[self.name.index(name)]
        return LinkPose(transform.xyz, transform.qua)


class JointState(cspace.cspace.classes.JointState):
    def __init__(self, position):
        self._position_ = torch.as_tensor(position, dtype=torch.float64)

    @property
    def position(self):
        return self._position_

    def transform(self, joint):
        def f_origin(position, xyz, rpy):
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

        def f_linear(position, axis, upper, lower):
            assert len(axis) == 3
            axis = [(index, sign) for index, sign in enumerate(axis) if sign]
            assert len(axis) == 1
            index, sign = next(iter(axis))
            assert (sign == 1) or (sign == -1)

            position = torch.clip(position, min=lower, max=upper)
            position = position if sign > 0 else -position

            zero = torch.zeros_like(position)

            if index == 0:
                xyz = torch.stack((position, zero, zero), dim=-1)
            elif index == 1:
                xyz = torch.stack((zero, position, zero), dim=-1)
            else:
                xyz = torch.stack((zero, zero, position), dim=-1)

            rot = torch.eye(
                3,
                device=position.device,
                dtype=position.dtype,
            ).expand(*(xyz.shape[:-1] + tuple([-1, -1])))

            return Transform(xyz=xyz, rot=rot)

        def f_angular(position, axis, upper, lower):
            assert len(axis) == 3
            axis = [(index, sign) for index, sign in enumerate(axis) if sign]
            assert len(axis) == 1
            index, sign = next(iter(axis))
            assert (sign == 1) or (sign == -1)

            position = (
                torch.clip(position, min=lower, max=upper)
                if (upper is not None or lower is not None)
                else position
            )
            position = position if sign > 0 else -position

            zero = torch.zeros_like(position)
            one = torch.ones_like(position)
            sin = torch.sin(position)
            cos = torch.cos(position)

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
                device=position.device,
                dtype=position.dtype,
            ).expand(*(rot.shape[:-2] + tuple([-1])))

            return Transform(xyz=xyz, rot=rot)

        e_transform = f_origin(
            self.position,
            joint.origin.xyz,
            joint.origin.rpy,
        )
        if joint.type == "fixed":
            return e_transform
        elif joint.type == "revolute":
            return e_transform * f_angular(
                self.position,
                joint.axis,
                joint.limit.upper,
                joint.limit.lower,
            )
        elif joint.type == "continuous":
            return e_transform * f_angular(
                self.position,
                joint.axis,
                None,
                None,
            )
        elif joint.type == "prismatic":
            return e_transform * f_linear(
                self.position,
                joint.axis,
                joint.limit.upper,
                joint.limit.lower,
            )
        raise NotImplementedError


class JointStateCollection(cspace.cspace.classes.JointStateCollection):
    def __init__(self, name, position):
        self._name_ = tuple(name)
        self._position_ = torch.as_tensor(position, dtype=torch.float64)
        assert len(self._name_) == self._position_.shape[-1]

    @property
    def name(self):
        return self._name_

    def __call__(self, name):
        if name not in self.name:
            return JointState(
                torch.empty(
                    self._position_.shape[:-1],
                    device=self._position_.device,
                    dtype=self._position_.dtype,
                )
            )
        index = self.name.index(name)
        return JointState(torch.select(self._position_, dim=-1, index=index))

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
        return LinkPoseCollection(
            link, tuple(self.transform(spec, item, base) for item in link)
        )


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
