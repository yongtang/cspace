import cspace.cspace.classes
import cspace.torch.ops
import dataclasses
import functools
import operator
import torch


@dataclasses.dataclass(frozen=True, kw_only=True)
class Transform:
    data: torch.Tensor

    @property
    def xyz(self):
        return self.data[..., :3]

    @property
    def qua(self):
        return self.data[..., 3:]

    @property
    def rpy(self):
        return cspace.torch.ops.qua_to_rpy(self.qua)

    @property
    def rot(self):
        return cspace.torch.ops.qua_to_rot(self.qua)

    def inverse(self):
        assert self.rot.ndim == 2 and other.rot.ndim == 2
        xyz = torch.mm(self.rot.transpose(), self.xyz)
        qua = cspace.torch.ops.rot_to_qua(self.rot.transpose())
        data = torch.concatenate((xyz, qua), dim=-1)
        return Transform(data=data)

    def __mul__(self, other):
        assert self.rot.ndim == 2 and other.rot.ndim == 2
        xyz = torch.mm(self.rot, other.xyz.unsqueeze(-1)).squeeze(-1) + self.xyz
        qua = cspace.torch.ops.rot_to_qua(torch.mm(self.rot, other.rot))
        data = torch.concatenate((xyz, qua), dim=-1)
        return Transform(data=data)


@dataclasses.dataclass(frozen=True, kw_only=True)
class Spec(cspace.cspace.classes.Spec):
    def __post_init__(self):
        super().__post_init__()

        def f_joint(joint, data):
            data = torch.as_tensor(data, dtype=torch.float64)
            data = data.unsqueeze(-1)

            shape = data.shape

            xyz = torch.as_tensor(
                joint.origin.xyz,
                device=data.device,
                dtype=torch.float64,
            )
            rpy = torch.as_tensor(
                joint.origin.rpy,
                device=data.device,
                dtype=torch.float64,
            )
            qua = cspace.torch.ops.rpy_to_qua(rpy)

            transform = Transform(data=torch.concatenate((xyz, qua), dim=-1))

            if isinstance(
                joint,
                (
                    cspace.cspace.classes.Revolute,
                    cspace.cspace.classes.Prismatic,
                    cspace.cspace.classes.Continuous,
                ),
            ):
                if isinstance(
                    joint,
                    (
                        cspace.cspace.classes.Revolute,
                        cspace.cspace.classes.Prismatic,
                    ),
                ):
                    data = torch.clip(
                        data,
                        min=joint.limit.lower,
                        max=joint.limit.upper,
                    )

                axis = torch.as_tensor(
                    joint.axis,
                    device=data.device,
                    dtype=torch.float64,
                )
                data = torch.multiply(data, axis)
                if isinstance(
                    joint,
                    (
                        cspace.cspace.classes.Revolute,
                        cspace.cspace.classes.Continuous,
                    ),
                ):
                    xyz = torch.as_tensor(
                        (0.0, 0.0, 0.0),
                        device=data.device,
                        dtype=torch.float64,
                    ).repeat(*shape)
                    qua = cspace.torch.ops.rpy_to_qua(data)
                else:
                    xyz = data
                    qua = torch.as_tensor(
                        (0.0, 0.0, 0.0, 1.0),
                        device=data.device,
                        dtype=torch.float64,
                    ).repeat(*shape)
                transform = transform * Transform(
                    data=torch.concatenate((xyz, qua), dim=-1)
                )

            return transform

        def f_link(spec, data, link, base=None):
            def f_transform(spec, data, name, forward):
                index = spec.joint.index(name)
                value = torch.select(data, dim=-1, index=index)
                transform = spec.joint(name).transform(value)

                transform = transform if forward else transform.inverse()

                return transform

            return functools.reduce(
                operator.mul,
                [
                    f_transform(spec, data, name, forward)
                    for name, forward in reversed(spec.route(link, base))
                ],
                Transform(
                    data=torch.as_tensor(
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                        dtype=torch.float64,
                    )
                ),
            )

        def f_spec(spec, data, *link, base=None):
            data = torch.as_tensor(data, dtype=torch.float64)
            assert data.shape[-1] == len(spec.joint)
            base = base if base else next(iter(spec.chain))[0]

            return torch.stack(
                tuple(f_link(spec, data, item, base=base).data for item in link), dim=-2
            )

        for joint in self.joint:
            object.__setattr__(joint, "function", f_joint)
        object.__setattr__(self, "function", f_spec)
