import cspace.cspace.classes
import cspace.torch.classes
import dataclasses
import functools
import itertools
import typing
import torch
import abc


@dataclasses.dataclass(frozen=True, kw_only=True)
class DataIndex:
    e: str | None
    name: str | None
    field: int | None


class DataBlock(abc.ABC):
    def __init__(self, dimension):
        self.dimension = int(dimension)

    def encode(self, data):
        value = torch.as_tensor(data, dtype=torch.float64)
        value = torch.special.expit(value)
        value = torch.clip(
            (value * (self.dimension - 1)).to(torch.int64),
            min=0,
            max=(self.dimension - 1),
        )
        return value

    def decode(self, data):
        value = torch.as_tensor(data, dtype=torch.int64)
        value = torch.clip(
            (data % self.dimension).to(torch.float64) / (self.dimension - 1),
            min=0.0,
            max=1.0,
        )
        value = torch.special.logit(value)
        return value


@dataclasses.dataclass(frozen=True, init=False)
class NoneIndex(DataIndex):
    def __init__(self):
        object.__setattr__(self, "e", None)
        object.__setattr__(self, "name", None)
        object.__setattr__(self, "field", None)


@dataclasses.dataclass(frozen=True, init=False)
class PoseIndex(DataIndex):
    def __init__(self, name, field):
        object.__setattr__(self, "e", "pose")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "field", field)


@dataclasses.dataclass(frozen=True, init=False)
class JointIndex(DataIndex):
    def __init__(self, name):
        object.__setattr__(self, "e", "joint")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "field", None)


class BlockEncoding(abc.ABC):
    def __init__(self, spec, link):
        self.blocks = []
        for name in link:
            for field in range(6):
                self.blocks.append(
                    (PoseIndex(name=name, field=field), DataBlock(dimension=10000))
                )
        for joint in spec.joint:
            self.blocks.append(
                (JointIndex(name=joint.name), DataBlock(dimension=10000))
            )
        self.blocks.append((NoneIndex(), DataBlock(dimension=1)))

    def encode(self, data):
        index, value = data
        index = self.indices.index(index)
        count = sum(map(lambda i: self.entries[i].dimension, range(index)))
        block = self.entries[index]
        return count + block.encode(value)

    def decode(self, data):
        count = 0
        for index, block in self.blocks:
            if count <= data and data < count + block.dimension:
                return index, block.decode(data - count)
            count += block.dimension
        raise ValueError(f"{data}")

    @functools.cached_property
    def indices(self):
        return tuple(index for index, block in self.blocks)

    @functools.cached_property
    def entries(self):
        return tuple(block for index, block in self.blocks)


class Kinematics:
    spec: cspace.cspace.classes.Spec
    base: str
    link: tuple[str]
    model: typing.Callable

    def __init__(self, description, *link, base=None):
        spec = cspace.cspace.classes.Spec(description=description)
        assert (not base) or (base in spec.link)
        base = str(base) if base else spec.base
        assert (not link) or all([(item in spec.link) for item in link])
        link = tuple(link) if link else spec.link
        self.spec = spec
        self.base = base
        self.link = link

    def forward(self, state):
        return state.forward(self.spec, *self.link, base=self.base)

    def inverse(self, pose):
        raise NotImplementedError

    def tokenize(self, data):
        def f_pose(spec, link, base, zero, link_transforms):
            zero_transform = zero.transform(spec, link, base)
            link_transform = link_transforms[link]
            transform = zero_transform.inverse() * link_transform
            return [
                self.encoding.encode((PoseIndex(name=link, field=field), entry))
                for field, entry in enumerate(
                    cspace.torch.ops.se3_log(transform.xyz, transform.rot)
                )
            ]

        name = tuple(joint.name for joint in self.spec.joint if joint.motion.call)
        zero = cspace.torch.classes.JointStateCollection(name, tuple(0.0 for e in name))

        if isinstance(data, cspace.cspace.classes.JointStateCollection):
            assert tuple(sorted(data.name)) == tuple(sorted(name))
            link_transforms = {
                link: data.transform(self.spec, link, self.base) for link in self.link
            }
            state = data
        elif isinstance(data, cspace.cspace.classes.LinkPoseCollection):
            assert False
        else:
            raise ValueError(f"{data}")

        entries = tuple(
            itertools.chain.from_iterable(
                tuple(
                    f_pose(self.spec, link, self.base, zero, link_transforms)
                    for link in self.link
                )
            )
        )
        entries = entries + tuple([self.encoding.encode((NoneIndex(), 0.0))])

        if state is not None:
            entries = entries + tuple(
                self.encoding.encode((JointIndex(name=name), state(name).position))
                for name in state.name
            )
            entries = entries + tuple([self.encoding.encode((NoneIndex(), 0.0))])
        return entries

    def assembly(self, data):
        entries = tuple(map(self.encoding.decode, data))
        index = list(index for index, entry in entries).index(NoneIndex())
        poses = entries[:index]
        entries = entries[index + 1 :]
        index = list(index for index, entry in entries).index(NoneIndex())
        states = entries[:index]

        name = tuple(joint.name for joint in self.spec.joint if joint.motion.call)
        position = [0.0 for e in name]
        for index, entry in states:
            if index.e == "joint":
                position[name.index(index.name)] += entry
        state = cspace.torch.classes.JointStateCollection(name, position)

        return state

    @property
    def encoding(self):
        return BlockEncoding(self.spec, self.link)
