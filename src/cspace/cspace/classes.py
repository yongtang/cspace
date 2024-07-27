import abc
import math
import typing
import operator
import itertools
import functools
import dataclasses
import xml.dom.minidom


class LinkPoseCollection(abc.ABC):
    def __init__(self, base, name):
        self._base_ = base
        self._name_ = tuple(name)

    @property
    def base(self):
        return self._base_

    @property
    def name(self):
        return self._name_

    @abc.abstractmethod
    def position(self, name):
        raise NotImplementedError

    @abc.abstractmethod
    def orientation(self, name):
        raise NotImplementedError

    @abc.abstractmethod
    def transform(self, name):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def batch(self):
        raise NotImplementedError


class JointStateCollection(abc.ABC):
    def __init__(self, name):
        self._name_ = tuple(name)

    @property
    def name(self):
        return self._name_

    @abc.abstractmethod
    def position(self, spec, name):
        raise NotImplementedError

    @abc.abstractmethod
    def forward(self, spec, *link, base=None):
        raise NotImplementedError

    @functools.lru_cache
    def joint(self, spec, name):
        lookup = spec.joint(name)
        origin = self.origin(
            self.position(spec, name),
            lookup.origin.xyz,
            lookup.origin.rpy,
        )
        if lookup.motion.call == "":
            return origin
        elif lookup.motion.call == "linear":
            return origin * self.linear(
                self.position(spec, name),
                lookup.motion.sign,
                lookup.motion.axis,
            )
        elif lookup.motion.call == "angular":
            return origin * self.angular(
                self.position(spec, name),
                lookup.motion.sign,
                lookup.motion.axis,
            )
        raise NotImplementedError

    def transform(self, spec, link, base):
        def f_transform(spec, name, forward):
            e_transform = self.joint(spec, name)
            e_transform = e_transform if forward else e_transform.inverse()
            return e_transform

        return functools.reduce(
            operator.mul,
            [
                f_transform(spec, name, forward)
                for name, forward in reversed(spec.route(link, base))
            ],
            self.identity(),
        )

    @property
    @abc.abstractmethod
    def batch(self):
        raise NotImplementedError

    @abc.abstractmethod
    def scale(self, spec, min, max):
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def apply(cls, spec, joint, scale, min, max):
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def identity(cls):
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def origin(cls, position, xyz, rpy):
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def linear(cls, position, sign, axis):
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def angular(cls, position, sign, axis):
        raise NotImplementedError

    @classmethod
    def angle(cls, value, zero, limit):
        return (value + limit) % (limit * 2.0) - limit


@dataclasses.dataclass(frozen=True, kw_only=True)
class Transform(abc.ABC):
    xyz: typing.Any
    rot: typing.Any

    @property
    @abc.abstractmethod
    def rpy(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def qua(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def log(self):
        raise NotImplementedError

    @abc.abstractmethod
    def inverse(self):
        raise NotImplementedError

    @abc.abstractmethod
    def __mul__(self, other):
        raise NotImplementedError


class Attribute:
    class Origin(tuple):
        def __new__(cls, items):
            items = tuple(float(item) for item in items)
            assert len(items) == 6
            return super().__new__(cls, items)

        @functools.cached_property
        def xyz(self):
            return self[:3]

        @functools.cached_property
        def rpy(self):
            return self[3:]

    @dataclasses.dataclass(init=False, frozen=True)
    class Motion:
        call: str
        sign: int
        axis: int
        zero: float
        limit: float

        def __init__(self, joint, axis, lower, upper):

            if joint == "fixed":
                call = ""
                lower, upper = 0.0, 0.0
            elif joint == "prismatic":
                call = "linear"
                lower, upper = float(lower), float(upper)
            elif joint == "revolute":
                call = "angular"
                lower, upper = float(lower), float(upper)
            elif joint == "continuous":
                call = "angular"
                lower, upper = -math.pi, math.pi
            else:
                raise NotImplementedError

            assert lower <= upper
            zero = (upper + lower) / 2.0
            limit = (upper - lower) / 2.0
            object.__setattr__(self, "call", call)
            object.__setattr__(self, "zero", zero)
            object.__setattr__(self, "limit", limit)

            axis = (1, 0, 0) if joint == "fixed" else axis
            axis = tuple(int(item) for item in axis)
            assert len(axis) == 3
            sign = tuple(item for item in axis if item)
            assert len(sign) == 1
            sign = next(iter(sign))
            assert (sign == 1) or (sign == -1)
            axis = axis.index(sign)
            object.__setattr__(self, "sign", sign)
            object.__setattr__(self, "axis", axis)


@dataclasses.dataclass(frozen=True, kw_only=True)
class Mimic:
    joint: str
    offset: float
    multiplier: float

    def __post_init__(self):
        assert self.joint

        offset = float(self.offset) if self.offset else 0.0
        multiplier = float(self.multiplier) if self.multiplier else 1.0

        object.__setattr__(self, "offset", offset)
        object.__setattr__(self, "multiplier", multiplier)


@dataclasses.dataclass(frozen=True, kw_only=True)
class Joint(abc.ABC):
    name: str
    child: str
    parent: str
    origin: Attribute.Origin
    motion: Attribute.Motion

    def __post_init__(self):
        pass


class JointCollection(tuple):
    def __new__(cls, items):
        items = tuple(item for item in items)
        assert all(
            map(
                lambda item: isinstance(item, Joint),
                items,
            )
        ), f"{items}"
        return super().__new__(cls, items)

    @functools.cache
    def index(self, name):
        def f_name(self):
            return list(item.name for item in self)

        return f_name(self).index(name)

    @functools.cache
    def __call__(self, name):
        return next(filter(lambda item: item.name == name, self))


class Chain(tuple):
    def __new__(cls, items):
        items = tuple(item for item in items)
        assert all(
            map(
                lambda item: isinstance(item, str),
                items,
            )
        ), f"{items}"
        return super().__new__(cls, items)


class ChainCollection(tuple):
    def __new__(cls, items):
        items = tuple(item for item in items)
        assert all(
            map(
                lambda item: isinstance(item, Chain),
                items,
            )
        ), f"{items}"
        return super().__new__(cls, items)

    def __call__(self, name):
        return next(filter(lambda item: item[-1] == name, self))


@dataclasses.dataclass(frozen=True, kw_only=True)
class Spec:
    description: str = dataclasses.field(repr=False)
    chain: ChainCollection = dataclasses.field(init=False)
    joint: JointCollection = dataclasses.field(init=False)

    def __post_init__(self):
        def f_attribute(e, name):
            entry = e.getAttribute(name)
            assert entry
            return entry

        def f_element(e, name):
            entries = e.getElementsByTagName(name)
            assert entries.length == 1
            entry = entries.item(0)
            assert entry
            return entry

        def f_origin(e):
            entries = e.getElementsByTagName("origin")
            assert entries.length == 0 or entries.length == 1
            xyz = (
                "0 0 0" if entries.length == 0 else entries.item(0).getAttribute("xyz")
            )
            rpy = (
                "0 0 0" if entries.length == 0 else entries.item(0).getAttribute("rpy")
            )
            xyz = xyz if xyz else "0 0 0"
            rpy = rpy if rpy else "0 0 0"
            xyz = tuple(float(e) for e in xyz.split(" "))
            rpy = tuple(float(e) for e in rpy.split(" "))
            assert len(xyz) == 3 and len(rpy) == 3
            origin = xyz + rpy
            return Attribute.Origin(origin)

        def f_motion(e):
            entries = e.getElementsByTagName("axis")
            assert entries.length == 0 or entries.length == 1
            xyz = (
                "1 0 0" if entries.length == 0 else entries.item(0).getAttribute("xyz")
            )
            entries = e.getElementsByTagName("limit")
            assert entries.length == 0 or entries.length == 1
            lower = (
                "0" if entries.length == 0 else entries.item(0).getAttribute("lower")
            )
            upper = (
                "0" if entries.length == 0 else entries.item(0).getAttribute("upper")
            )
            return Attribute.Motion(
                f_attribute(e, "type"), xyz.split(" "), lower, upper
            )

        def f_mimic(e, joint):
            entries = e.getElementsByTagName("mimic")
            assert entries.length == 0 or entries.length == 1
            if entries.length == 0:
                return None
            assert (
                joint(f_attribute(e, "name")).motion.call
                and joint(f_attribute(e, "name")).motion.call
                == joint(entries.item(0).getAttribute("joint")).motion.call
            )

            offset = entries.item(0).getAttribute("offset")
            multiplier = entries.item(0).getAttribute("multiplier")

            offset = float(offset) if offset else 0.0
            multiplier = float(multiplier) if multiplier else 1.0

            lower = (
                joint(entries.item(0).getAttribute("joint")).motion.zero
                - joint(entries.item(0).getAttribute("joint")).motion.limit
            )
            upper = (
                joint(entries.item(0).getAttribute("joint")).motion.zero
                + joint(entries.item(0).getAttribute("joint")).motion.limit
            )

            lower = lower * multiplier + offset
            upper = upper * multiplier + offset

            lower, upper = (lower, upper) if (lower < upper) else (upper, lower)

            assert lower >= (
                joint(f_attribute(e, "name")).motion.zero
                - joint(f_attribute(e, "name")).motion.limit
            )
            assert upper <= (
                joint(f_attribute(e, "name")).motion.zero
                + joint(f_attribute(e, "name")).motion.limit
            )

            return Mimic(
                joint=entries.item(0).getAttribute("joint"),
                offset=offset,
                multiplier=multiplier,
            )

        def f_joint(e):
            name = f_attribute(e, "name")
            child = f_attribute(f_element(e, "child"), "link")
            parent = f_attribute(f_element(e, "parent"), "link")
            origin = f_origin(e)
            motion = f_motion(e)
            return Joint(
                name=name,
                child=child,
                parent=parent,
                origin=origin,
                motion=motion,
            )

        def f_chain(entry, joint):
            chain = [entry]
            while True:
                lookup = [e.parent for e in joint if e.child == next(iter(chain))]
                if len(lookup) == 0:
                    break
                assert len(lookup) == 1
                lookup = next(iter(lookup))
                assert lookup not in chain
                chain = [lookup] + chain
            return chain

        with xml.dom.minidom.parseString(self.description) as dom:
            joint = JointCollection(
                tuple(
                    f_joint(e)
                    for e in dom.getElementsByTagName("joint")
                    if e.parentNode == dom.documentElement
                )
            )
            assert len(joint) == len(
                list(
                    e
                    for e in dom.getElementsByTagName("joint")
                    if e.parentNode == dom.documentElement
                )
            )

            link = {f_attribute(e, "name") for e in dom.getElementsByTagName("link")}
            assert len(link) == dom.getElementsByTagName("link").length

            assert link == set(
                itertools.chain.from_iterable(map(lambda e: [e.child, e.parent], joint))
            )

            chain = ChainCollection([Chain(f_chain(e, joint)) for e in sorted(link)])
            assert len({next(iter(e)) for e in chain}) == 1

            mimic = {
                name: entry
                for name, entry in map(
                    lambda e: (f_attribute(e, "name"), f_mimic(e, joint)),
                    dom.getElementsByTagName("joint"),
                )
                if entry
            }
            object.__setattr__(self, "joint", joint)
            object.__setattr__(self, "chain", chain)
            object.__setattr__(self, "mimic", mimic)

    @functools.lru_cache
    def f_route(self, source, target):
        source_chain = tuple(
            reversed(next(filter(lambda item: item[-1] == source, self.chain)))
        )
        target_chain = next(filter(lambda item: item[-1] == target, self.chain))[1:]
        return source_chain + target_chain

    @functools.lru_cache
    def f_lookup(self, source, target):
        for item in self.joint:
            if item.parent == source and item.child == target:
                return (item.name, False)
            elif item.parent == target and item.child == source:
                return (item.name, True)

    @functools.cache
    def route(self, source, target=None):

        target = target if target else self.base

        route_chain = self.f_route(source, target)

        route_forward = route_chain[: route_chain.index(target) + 1]
        route_inverse = route_chain[
            len(route_chain) - 1 - list(reversed(route_chain)).index(source) :
        ]
        route_final = (
            route_forward if len(route_forward) <= len(route_inverse) else route_inverse
        )

        return [
            self.f_lookup(route_final[i - 1], route_final[i])
            for i in range(1, len(route_final))
        ]

    @functools.cached_property
    def link(self):
        return tuple(item[-1] for item in self.chain)

    @functools.cached_property
    def base(self):
        return next(iter(next(iter(self.chain))))

    def forward(self, state, *link, base=None):
        base = base if base else self.base
        link = link if len(link) else self.link
        return state.forward(self, *link, base=base)


class Kinematics:
    spec: Spec
    base: str
    link: tuple[str]
    joint: tuple[str]

    def __init__(self, description, *link, base=None):
        spec = Spec(description=description)
        assert (not base) or (base in spec.link)
        base = str(base) if base else spec.base
        assert (not link) or all([(item in spec.link) for item in link])
        link = tuple(link) if link else spec.link

        joint = list([e for e, _ in spec.route(e, base)] for e in link)
        joint = list(set(itertools.chain.from_iterable(joint)))
        joint = tuple(
            e
            for e in sorted(joint)
            if (spec.joint(e).motion.call and e not in spec.mimic)
        )

        self.spec = spec
        self.base = base
        self.link = link
        self.joint = joint

    def forward(self, state):
        return state.forward(self.spec, *self.link, base=self.base)

    def inverse(self, pose):
        raise NotImplementedError

    def policy(self, state, observation):
        raise NotImplementedError
