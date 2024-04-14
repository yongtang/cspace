import abc
import dataclasses
import xml.dom.minidom


@dataclasses.dataclass(init=False, frozen=True, repr=False)
class Joint(abc.ABC):
    name: str
    child: str
    parent: str
    origin: tuple[float, float, float, float, float, float]

    def __repr__(self):
        raise NotImplementedError

    @dataclasses.dataclass(kw_only=True, frozen=True, repr=False)
    class Limit:
        lower: float
        upper: float
        effort: float
        velocity: float

        def __repr__(self):
            return f"(lower={self.lower}, upper={self.upper}, effort={self.effort}, velocity={self.velocity})"

    class Collection(tuple):
        def __new__(cls, items):
            items = tuple([item for item in items])
            assert all(
                map(
                    lambda e: isinstance(e, Joint),
                    items,
                )
            ), f"{items}"
            return super().__new__(cls, items)

        def __call__(self, name):
            return next(filter(lambda item: item.name == name, self))


@dataclasses.dataclass(kw_only=True, frozen=True)
class Fixed(Joint):
    def __post_init__(self):
        pass


@dataclasses.dataclass(kw_only=True, frozen=True)
class Revolute(Joint):
    axis: tuple[int, int, int]
    limit: Joint.Limit

    def __post_init__(self):
        assert self.limit is not None


@dataclasses.dataclass(kw_only=True, frozen=True)
class Continuous(Joint):
    axis: tuple[int, int, int]

    def __post_init__(self):
        pass


@dataclasses.dataclass(kw_only=True, frozen=True)
class Prismatic(Joint):
    axis: tuple[int, int, int]
    limit: Joint.Limit

    def __post_init__(self):
        assert self.limit is not None


@dataclasses.dataclass(frozen=True, kw_only=True)
class Description:
    description: str = dataclasses.field(repr=False)
    joint: tuple = dataclasses.field(init=False)

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
            xyz = [float(e) for e in xyz.split(" ")]
            rpy = [float(e) for e in rpy.split(" ")]
            assert len(xyz) == 3 and len(rpy) == 3
            return tuple(xyz + rpy)

        def f_axis(e):
            entries = e.getElementsByTagName("axis")
            assert entries.length == 0 or entries.length == 1
            xyz = (
                "1 0 0" if entries.length == 0 else entries.item(0).getAttribute("xyz")
            )
            xyz = xyz.split(" ")
            assert len(xyz) == 3
            assert xyz.count("0") == 2
            assert xyz.count("1") + xyz.count("-1") == 1
            xyz = tuple(int(e) for e in xyz)
            return xyz

        def f_limit(e):
            entries = e.getElementsByTagName("limit")
            assert entries.length == 0 or entries.length == 1
            if entries.length == 0:
                return None
            lower = entries.item(0).getAttribute("lower")
            upper = entries.item(0).getAttribute("upper")
            effort = entries.item(0).getAttribute("effort")
            velocity = entries.item(0).getAttribute("velocity")
            lower = float(lower if lower else "0")
            upper = float(upper if upper else "0")
            effort = float(effort)
            velocity = float(velocity)
            return Joint.Limit(
                lower=lower, upper=upper, effort=effort, velocity=velocity
            )

        def f_mimic(e):
            entries = e.getElementsByTagName("mimic")
            assert entries.length == 0, "TODO: mimic"
            return None

        def f_joint(e):
            name = f_attribute(e, "name")
            child = f_attribute(f_element(e, "child"), "link")
            parent = f_attribute(f_element(e, "parent"), "link")
            origin = f_origin(e)
            axis = f_axis(e)
            limit = f_limit(e)
            mimic = f_mimic(e)
            if f_attribute(e, "type") == "fixed":
                return Fixed(
                    name=name,
                    child=child,
                    parent=parent,
                    origin=origin,
                )
            elif f_attribute(e, "type") == "revolute":
                return Revolute(
                    name=name,
                    child=child,
                    parent=parent,
                    origin=origin,
                    axis=axis,
                    limit=limit,
                )
            elif f_attribute(e, "type") == "continuous":
                return Continuous(
                    name=name,
                    child=child,
                    parent=parent,
                    origin=origin,
                    axis=axis,
                )
            elif f_attribute(e, "type") == "prismatic":
                return Prismatic(
                    name=name,
                    child=child,
                    parent=parent,
                    origin=origin,
                    axis=axis,
                    limit=limit,
                )
            else:
                raise NotImplementedError(f_attribute(e, "type"))

        with xml.dom.minidom.parseString(self.description) as dom:
            joint = Joint.Collection(
                tuple(f_joint(e) for e in dom.getElementsByTagName("joint"))
            )
            object.__setattr__(self, "joint", joint)
