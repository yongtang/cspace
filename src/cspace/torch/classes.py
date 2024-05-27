import cspace.cspace.classes
import cspace.torch.ops
import dataclasses
import functools
import torch


@dataclasses.dataclass(frozen=True, kw_only=True)
class Transform:
    xyz: torch.Tensor
    qua: torch.Tensor

    @property
    def rpy(self):
        return cspace.torch.ops.qua_to_rpy(self.qua)

    def __mul__(self, other):
        xyz = self.xyz + other.xyz
        qua = cspace.torch.ops.qua_mul_qua(self.qua, other.qua)
        return Transform(xyz=xyz, qua=qua)


@dataclasses.dataclass(frozen=True, kw_only=True)
class Spec(cspace.cspace.classes.Spec):
    def __post_init__(self):
        super().__post_init__()

        def f_transform(joint, data):
            data = data.unsqueeze(-1)

            shape = data.shape

            xyz = torch.as_tensor(
                joint.origin.xyz,
                device=data.device,
            )
            rpy = torch.as_tensor(
                joint.origin.rpy,
                device=data.device,
            )
            qua = cspace.torch.ops.rpy_to_qua(rpy)

            transform = Transform(xyz=xyz.repeat(*shape), qua=qua.repeat(*shape))

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
                    ).repeat(*shape)
                    qua = cspace.torch.ops.rpy_to_qua(data)
                else:
                    xyz = data
                    qua = torch.as_tensor(
                        (0.0, 0.0, 0.0, 1.0),
                        device=data.device,
                    ).repeat(*shape)
                transform = transform * Transform(xyz=xyz, qua=qua)

            return transform

        for joint in self.joint:
            object.__setattr__(joint, "function", f_transform)
