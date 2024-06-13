import abc
import math
import typing
import operator
import itertools
import functools
import dataclasses
import xml.dom.minidom


@dataclasses.dataclass(frozen=True, kw_only=True)
class LinkPose(abc.ABC):
    base: str
    name: str
    position: typing.Any
    orientation: typing.Any


class LinkPoseCollection(abc.ABC):
    @property
    @abc.abstractmethod
    def base(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def name(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def position(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def orientation(self):
        raise NotImplementedError

    @abc.abstractmethod
    def __call__(self, name):
        raise NotImplementedError


@dataclasses.dataclass(frozen=True, kw_only=True)
class JointState(abc.ABC):
    name: str
    position: typing.Any

    def transform(self, spec):
        joint = spec.joint(self.name)

        origin = self.origin(
            self.position,
            joint.origin.xyz,
            joint.origin.rpy,
        )
        if joint.motion.call == "":
            return origin
        elif joint.motion.call == "linear":
            return origin * self.linear(
                self.position,
                joint.motion.sign,
                joint.motion.axis,
            )
        elif joint.motion.call == "angular":
            return origin * self.angular(
                self.position,
                joint.motion.sign,
                joint.motion.axis,
            )
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


class JointStateCollection(abc.ABC):
    @property
    @abc.abstractmethod
    def name(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def position(self):
        raise NotImplementedError

    @abc.abstractmethod
    def __call__(self, name):
        raise NotImplementedError

    @abc.abstractmethod
    def identity(self):
        raise NotImplementedError

    @abc.abstractmethod
    def forward(self, spec, *link, base=None):
        raise NotImplementedError

    def transform(self, spec, link, base):
        def f_transform(spec, name, forward):
            state = self.__call__(name)
            e_transform = state.transform(spec)
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


@dataclasses.dataclass(frozen=True, kw_only=True)
class Transform(abc.ABC):
    xyz: typing.Any
    rot: typing.Any

    @property
    def rpy(self):
        raise NotImplementedError

    @property
    def qua(self):
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

        def __init__(self, joint, axis):
            if joint == "fixed":
                call = ""
            elif joint == "prismatic":
                call = "linear"
            elif joint in ("revolute", "continuous"):
                call = "angular"
            else:
                raise NotImplementedError
            axis = tuple(int(item) for item in axis)
            assert len(axis) == 3
            sign = tuple(item for item in axis if item)
            assert len(sign) == 1
            sign = next(iter(sign))
            assert (sign == 1) or (sign == -1)
            axis = axis.index(sign)
            object.__setattr__(self, "call", call)
            object.__setattr__(self, "sign", sign)
            object.__setattr__(self, "axis", axis)

    @dataclasses.dataclass(init=False, frozen=True)
    class Limit:
        zero: float
        lower: float
        upper: float

        def __init__(self, joint, lower, upper):
            if joint in ("prismatic", "revolute"):
                lower = float(lower)
                upper = float(upper)
                zero = (lower + upper) / 2.0
            elif joint == "fixed":
                lower = 0.0
                upper = 0.0
                zero = 0.0
            elif joint == "continuous":
                lower = -math.pi
                upper = math.pi
                zero = 0.0
            else:
                raise NotImplementedError
            object.__setattr__(self, "zero", zero)
            object.__setattr__(self, "lower", lower)
            object.__setattr__(self, "upper", upper)


@dataclasses.dataclass(frozen=True, kw_only=True)
class Joint(abc.ABC):
    name: str
    child: str
    parent: str
    origin: Attribute.Origin
    motion: Attribute.Motion
    limit: Attribute.Limit

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
        @functools.cache
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
            return Attribute.Motion(f_attribute(e, "type"), xyz.split(" "))

        def f_limit(e):
            entries = e.getElementsByTagName("limit")
            assert entries.length == 0 or entries.length == 1
            lower = (
                "0" if entries.length == 0 else entries.item(0).getAttribute("lower")
            )
            upper = (
                "0" if entries.length == 0 else entries.item(0).getAttribute("upper")
            )
            return Attribute.Limit(f_attribute(e, "type"), lower, upper)

        def f_mimic(e):
            entries = e.getElementsByTagName("mimic")
            assert entries.length == 0, "TODO: mimic"
            return None

        def f_joint(e):
            name = f_attribute(e, "name")
            child = f_attribute(f_element(e, "child"), "link")
            parent = f_attribute(f_element(e, "parent"), "link")
            origin = f_origin(e)
            motion = f_motion(e)
            limit = f_limit(e)
            mimic = f_mimic(e)
            return Joint(
                name=name,
                child=child,
                parent=parent,
                origin=origin,
                motion=motion,
                limit=limit,
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
                tuple(f_joint(e) for e in dom.getElementsByTagName("joint"))
            )
            assert len(joint) == dom.getElementsByTagName("joint").length

            link = {f_attribute(e, "name") for e in dom.getElementsByTagName("link")}
            assert len(link) == dom.getElementsByTagName("link").length

            assert link == set(
                itertools.chain.from_iterable(map(lambda e: [e.child, e.parent], joint))
            )

            chain = ChainCollection([Chain(f_chain(e, joint)) for e in sorted(link)])
            assert len({next(iter(e)) for e in chain}) == 1

            object.__setattr__(self, "joint", joint)
            object.__setattr__(self, "chain", chain)

    @functools.cache
    def route(self, source, target=None):
        @functools.lru_cache
        def f_route(chain, source, target):
            source_chain = tuple(
                reversed(next(filter(lambda item: item[-1] == source, chain)))
            )
            target_chain = next(filter(lambda item: item[-1] == target, chain))[1:]
            return source_chain + target_chain

        @functools.lru_cache
        def f_lookup(joint, source, target):
            for item in joint:
                if item.parent == source and item.child == target:
                    return (item.name, False)
                elif item.parent == target and item.child == source:
                    return (item.name, True)

        target = target if target else self.base

        route_chain = f_route(self.chain, source, target)

        route_forward = route_chain[: route_chain.index(target) + 1]
        route_inverse = route_chain[
            len(route_chain) - 1 - list(reversed(route_chain)).index(source) :
        ]
        route_final = (
            route_forward if len(route_forward) <= len(route_inverse) else route_inverse
        )

        return [
            f_lookup(self.joint, route_final[i - 1], route_final[i])
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
