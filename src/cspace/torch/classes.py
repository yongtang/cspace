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

        @functools.cache
        def f_index(self, name):
            return self.name.index(name)

        self._f_index_ = functools.partial(f_index, self)

    def position(self, name):
        return torch.select(self._position_, dim=-1, index=self._f_index_(name))

    def orientation(self, name):
        return torch.select(self._orientation_, dim=-1, index=self._f_index_(name))

    def transform(self, name):
        return Transform(
            xyz=self.position(name),
            rot=cspace.torch.ops.qua_to_rot(self.orientation(name)),
        )

    @property
    def batch(self):
        return tuple(self._position_.shape[:-2])

    @classmethod
    def stack(cls, value):
        return torch.stack(value, dim=-1)

    @classmethod
    def scale(cls, value, limit):
        assert value.shape[-1] == 6

        linear = value[..., :3]
        angular = value[..., 3:]

        assert torch.all(-limit <= linear) and torch.all(
            linear <= limit
        ), "({}, {}) vs. ({}, {})".format(
            torch.min(linear), torch.max(linear), -limit, limit
        )
        linear = torch.clip(linear, min=-limit, max=limit)
        linear = (linear + limit) / (limit * 2.0)

        angular = (angular + torch.pi) % (torch.pi * 2.0) - torch.pi
        angular = torch.clip(angular, min=-torch.pi, max=torch.pi)
        angular = (angular + torch.pi) / (torch.pi * 2.0)

        value = torch.concatenate((linear, angular), dim=-1)
        value = torch.clip(value, min=0.0, max=1.0)

        return value


class JointStateCollection(cspace.cspace.classes.JointStateCollection):
    def __init__(self, name, position):
        super().__init__(name=name)
        self._position_ = torch.as_tensor(position, dtype=torch.float64)
        assert len(self.name) == self._position_.shape[-1]

    def position(self, spec, name):
        @functools.cache
        def f_index(self, name):
            return self.name.index(name)

        if not spec.joint(name).motion.call:  # fixed
            return torch.empty(
                self._position_.shape[:-1],
                device=self._position_.device,
                dtype=self._position_.dtype,
            )
        mimic = spec.mimic.get(name, None)
        index = f_index(self, mimic.joint if mimic else name)
        value = torch.select(self._position_, dim=-1, index=index)
        return (value * mimic.multiplier + mimic.offset) if mimic else value

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
                    self.position(spec, name),
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

            self_value = self.position(spec, name)
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

            other_value = other.position(spec, name)
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
        name = {collection.name for collection in collections}
        assert len(name) == 1
        name = next(iter(name))

        position = torch.stack(
            tuple(collection._position_ for collection in collections)
        )

        return cls(name=name, position=position)

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


class Dataset(torch.utils.data.Dataset):
    def __init__(self, pose, state):
        assert isinstance(pose, cspace.torch.classes.LinkPoseCollection)
        assert len(pose.batch) == 1
        assert isinstance(state, cspace.torch.classes.JointStateCollection)
        assert len(state.batch) == 1
        assert pose.batch == state.batch

        self._pose_ = pose
        self._state_ = state

    def __len__(self):
        return self._pose_.batch[0]

    def __getitem__(self, key):
        pose = cspace.torch.classes.LinkPoseCollection(
            base=self._pose_.base,
            name=self._pose_.name,
            position=torch.select(self._pose_._position_, dim=0, index=key),
            orientation=torch.select(self._pose_._orientation_, dim=0, index=key),
        )
        state = cspace.torch.classes.JointStateCollection(
            name=self._state_.name,
            position=torch.select(self._state_._position_, dim=0, index=key),
        )
        return pose, state

    @classmethod
    def collate_fn(cls, entries):
        pose, state = list(zip(*entries))

        info = {(e.base, e.name) for e in pose}
        assert len(info) == 1
        base, name = next(iter(info))
        position = torch.stack(tuple(e._position_ for e in pose), dim=0)
        orientation = torch.stack(tuple(e._orientation_ for e in pose), dim=0)

        pose = cspace.torch.classes.LinkPoseCollection(
            base=base,
            name=name,
            position=position,
            orientation=orientation,
        )

        info = {e.name for e in state}
        assert len(info) == 1
        name = next(iter(info))
        position = torch.stack(tuple(e._position_ for e in state), dim=0)

        state = cspace.torch.classes.JointStateCollection(
            name=name,
            position=position,
        )

        return pose, state


class Kinematics(cspace.cspace.classes.Kinematics):
    model: torch.nn.Module

    def __init__(self, description, *link, base=None, model=None):
        super().__init__(description, *link, base=base)
        self.model = model

    def inverse(self, pose):
        with torch.no_grad():
            data = self.encode(pose)

            pred = self.model(data)

            state = self.decode(pred)

            return state

    @classmethod
    def load(cls, file):
        return torch.load(file)

    def train(
        self, *, total=None, epoch=None, batch=None, noise=None, seed=None, save=None
    ):
        epoch = epoch if epoch else 1
        batch_size = batch if batch else 128
        entry_total = total if total else 1024

        optimizer = torch.optim.AdamW(self.model.parameters())
        scheduler = torch.optim.lr_scheduler.ChainedScheduler(
            [
                torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9),
                torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer, T_0=5, T_mult=2
                ),
            ]
        )

        dataset = Dataset(
            *self.rand(
                total=entry_total,
                noise=noise,
                seed=seed,
            )
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=cspace.torch.classes.Dataset.collate_fn,
            shuffle=True,
        )

        self.optimize(
            dataloader=dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            save=save,
        )

    def rand(self, *, total, noise, seed=None, std=0.01):
        generator = torch.Generator().manual_seed(seed)
        zero = cspace.torch.classes.JointStateCollection.zero(
            self.spec, self.joint, batch=[total]
        )
        state = zero.apply(
            self.spec,
            torch.rand(
                total, len(self.joint), generator=generator, dtype=torch.float64
            ),
        )
        pose = self.forward(state)

        if noise:
            linear_shape = [noise] + list(pose.batch) + [3, len(pose.name)]

            linear_std = torch.ones(linear_shape, dtype=torch.float64) * std
            linear_mean = torch.zeros(linear_shape, dtype=torch.float64)

            linear_noise = torch.normal(linear_mean, linear_std, generator=generator)

            position = pose._position_
            linear = position.expand(*linear_shape)
            linear = linear + linear_noise
            position = linear

            angular_shape = [noise] + list(pose.batch) + [3, len(pose.name)]

            angular_std = torch.ones(angular_shape, dtype=torch.float64) * std
            angular_mean = torch.zeros(angular_shape, dtype=torch.float64)

            angular_noise = torch.normal(angular_mean, angular_std, generator=generator)

            orientation = pose._orientation_
            angular = torch.transpose(
                cspace.torch.ops.qua_to_rpy(torch.transpose(orientation, -2, -1)),
                -2,
                -1,
            )
            angular = angular + angular_noise
            orientation = torch.transpose(
                cspace.torch.ops.rpy_to_qua(torch.transpose(angular, -2, -1)), -2, -1
            )

            pose = cspace.torch.classes.LinkPoseCollection(
                base=pose.base,
                name=pose.name,
                position=torch.flatten(position, 0, 1),
                orientation=torch.flatten(orientation, 0, 1),
            )

            shape = [noise] + list(state.batch) + [len(state.name)]

            position = state._position_
            position = position.expand(*shape)

            state = cspace.torch.classes.JointStateCollection(
                name=state.name,
                position=torch.flatten(position, 0, 1),
            )

        return pose, state

    def optimize(self, *, dataloader, optimizer, scheduler, epoch, save):
        raise NotImplementedError

    def encode(self, pose):
        raise NotImplementedError

    def decode(self, pred):
        raise NotImplementedError
