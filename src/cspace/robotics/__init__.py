import abc
import itertools
import dataclasses
import xml.dom.minidom
from . import classes


@dataclasses.dataclass(frozen=True, kw_only=True)
class Spec:
    description: str = dataclasses.field(repr=False)
    chain: list = dataclasses.field(init=False)
    joint: tuple = dataclasses.field(init=False)
    link: dict = dataclasses.field(init=False)

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
            return classes.Attribute.Limit(
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
                return classes.Fixed(
                    name=name,
                    child=child,
                    parent=parent,
                    origin=origin,
                )
            elif f_attribute(e, "type") == "revolute":
                return classes.Revolute(
                    name=name,
                    child=child,
                    parent=parent,
                    origin=origin,
                    axis=axis,
                    limit=limit,
                )
            elif f_attribute(e, "type") == "continuous":
                return classes.Continuous(
                    name=name,
                    child=child,
                    parent=parent,
                    origin=origin,
                    axis=axis,
                )
            elif f_attribute(e, "type") == "prismatic":
                return classes.Prismatic(
                    name=name,
                    child=child,
                    parent=parent,
                    origin=origin,
                    axis=axis,
                    limit=limit,
                )
            else:
                raise NotImplementedError(f_attribute(e, "type"))

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
            joint = classes.Collection(
                tuple(f_joint(e) for e in dom.getElementsByTagName("joint"))
            )
            assert len(joint) == dom.getElementsByTagName("joint").length

            link = {
                f_attribute(e, "name") for e in dom.getElementsByTagName("link")
            }
            assert len(link) == dom.getElementsByTagName("link").length

            assert link == set(
                itertools.chain.from_iterable(map(lambda e: [e.child, e.parent], joint))
            )

            chain = [f_chain(e, joint) for e in link]
            assert len({next(iter(e)) for e in chain}) == 1

            object.__setattr__(self, "chain", chain)
            object.__setattr__(self, "joint", joint)
            object.__setattr__(self, "link", link)
