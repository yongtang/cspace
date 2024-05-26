import abc
import dataclasses


class Attribute:
    @dataclasses.dataclass(kw_only=True, frozen=True, repr=False)
    class Limit:
        lower: float
        upper: float
        effort: float
        velocity: float

        def __repr__(self):
            return f"(lower={self.lower}, upper={self.upper}, effort={self.effort}, velocity={self.velocity})"

    @dataclasses.dataclass(kw_only=True, frozen=True, repr=False)
    class Origin:
        xyz: [float, float, float]
        rpy: [float, float, float]

        def __repr__(self):
            return f"(xyz={self.xyz}, rpy={self.rpy})"


@dataclasses.dataclass(init=False, frozen=True, repr=False)
class Joint(abc.ABC):
    name: str
    child: str
    parent: str
    origin: Attribute.Origin

    def __repr__(self):
        raise NotImplementedError


@dataclasses.dataclass(kw_only=True, frozen=True)
class Fixed(Joint):
    def __post_init__(self):
        pass


@dataclasses.dataclass(kw_only=True, frozen=True)
class Revolute(Joint):
    axis: tuple[int, int, int]
    limit: Attribute.Limit

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
    limit: Attribute.Limit

    def __post_init__(self):
        assert self.limit is not None


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
        return next(filter(lambda item: item.name == name, self))
