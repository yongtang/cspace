import cspace.cspace.classes
import cspace.torch.classes
import dataclasses
import itertools
import typing
import math
import abc


class DataEncoding(abc.ABC):
    @dataclasses.dataclass(frozen=True, kw_only=True)
    class ScaleRecord:
        e: str | None
        name: str | None
        index: int | None
        entry: float | None

    @dataclasses.dataclass(frozen=True, kw_only=True)
    class ScaleEncoding:
        scale: int

        def __post_init__(self):
            assert self.scale >= 0

        def encode(self, data):
            if self.scale == 0:
                assert data is None
                return 0

            assert -1.0 <= data and data <= 1.0
            entry = int(data * self.scale)
            return min(max(entry, -self.scale), self.scale) + self.scale

        def decode(self, data):
            if self.scale == 0:
                assert data == 0
                return None

            assert 0 <= data and data <= self.scale * 2
            entry = float((data - self.scale) / self.scale)
            return min(max(entry, -1.0), 1.0)

        @property
        def dimension(self):
            return self.scale * 2 + 1

    def __init__(self, spec, link):
        self.link = link
        self.joint = tuple(joint.name for joint in spec.joint if joint.motion.call)

        self.link_index = 6
        self.link_scale = 100
        self.joint_scale = 100

        self.bucket = []
        for link in self.link:
            for index in range(self.link_index):
                lookup = DataEncoding.ScaleRecord(
                    e="link", name=link, index=index, entry=None
                )
                encoding = DataEncoding.ScaleEncoding(scale=self.link_scale)
                self.bucket.append((lookup, encoding))
        for joint in self.joint:
            lookup = DataEncoding.ScaleRecord(
                e="joint", name=joint, index=0, entry=None
            )
            encoding = DataEncoding.ScaleEncoding(scale=self.joint_scale)
            self.bucket.append((lookup, encoding))

        self.bucket.append(
            (
                DataEncoding.ScaleRecord(e=None, name=None, index=None, entry=None),
                DataEncoding.ScaleEncoding(scale=0),
            )
        )

    def encode(self, data):
        index = 0
        for lookup, encoding in self.bucket:
            if lookup == DataEncoding.ScaleRecord(
                e=data.e, name=data.name, index=data.index, entry=None
            ):
                return index + encoding.encode(data.entry)
            index += encoding.dimension

        raise ValueError("{}".format(data))

    def decode(self, data):
        index = 0
        for lookup, encoding in self.bucket:
            if index <= data and data < index + encoding.dimension:
                entry = encoding.decode(data - index)
                return DataEncoding.ScaleRecord(
                    e=lookup.e, name=lookup.name, index=lookup.index, entry=entry
                )
            index += encoding.dimension

        raise ValueError("{}".format(data))


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
        if isinstance(data, cspace.cspace.classes.LinkPoseCollection):
            assert False

        state = data

        entries = tuple(
            DataEncoding.ScaleRecord(
                e="joint", name=name, index=0, entry=(state(name).position)
            )
            for name in state.name
        )
        count = math.ceil(max(abs(entry.entry) for entry in entries))
        entries = tuple(
            [
                DataEncoding.ScaleRecord(
                    e=entry.e,
                    name=entry.name,
                    index=entry.index,
                    entry=(entry.entry / count if count > 0 else 0.0),
                )
            ]
            * count
            for entry in entries
        )
        entries = tuple(
            entry
            for entry in itertools.chain.from_iterable(zip(*entries))
            if abs(entry.entry) > 0.0
        )

        def f_link(spec, link, base, zero):
            zero_transform = zero.transform(spec, link, base)
            link_transform = state.transform(spec, link, base)
            transform = zero_transform.inverse() * link_transform
            return [
                DataEncoding.ScaleRecord(e="link", name=link, index=index, entry=entry)
                for index, entry in enumerate(
                    cspace.torch.ops.se3_log(transform.xyz, transform.rot)
                )
            ]

        zero = cspace.torch.classes.JointStateCollection(
            state.name, tuple(0.0 for name in state.name)
        )
        links = tuple(
            itertools.chain.from_iterable(
                tuple(f_link(self.spec, link, self.base, zero) for link in self.link)
            )
        )
        count = math.ceil(max(abs(entry.entry) for entry in links))
        links = tuple(
            [
                DataEncoding.ScaleRecord(
                    e=entry.e,
                    name=entry.name,
                    index=entry.index,
                    entry=(entry.entry / count if count > 0 else 0.0),
                )
            ]
            * count
            for entry in links
        )
        links = tuple(
            entry
            for entry in itertools.chain.from_iterable(zip(*links))
            if abs(entry.entry) > 0.0
        )

        encoding = DataEncoding(self.spec, self.link)

        none = tuple(
            [DataEncoding.ScaleRecord(e=None, name=None, index=None, entry=None)]
        )

        return tuple(map(encoding.encode, links + none + entries + none))

    def assembly(self, data):
        encoding = DataEncoding(self.spec, self.link)
        entries = tuple(map(encoding.decode, data))
        index = entries.index(
            DataEncoding.ScaleRecord(e=None, name=None, index=None, entry=None)
        )
        entries = entries[index + 1 :]
        index = entries.index(
            DataEncoding.ScaleRecord(e=None, name=None, index=None, entry=None)
        )
        entries = entries[:index]

        entries = tuple(entry for entry in entries if entry.e == "joint")

        name = tuple(joint.name for joint in self.spec.joint if joint.motion.call)
        position = [0.0 for k in name]
        for entry in entries:
            position[name.index(entry.name)] += entry.entry

        state = cspace.torch.classes.JointStateCollection(name, position)

        return state
