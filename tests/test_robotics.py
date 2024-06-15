import cspace.cspace.classes

import pathlib
import logging


def test_spec(device, urdf_file_tutorial):
    spec = cspace.cspace.classes.Spec(
        description=pathlib.Path(urdf_file_tutorial).read_text()
    )

    joints = {
        "base_to_right_leg": cspace.cspace.classes.Joint(
            name="base_to_right_leg",
            child="right_leg",
            parent="base_link",
            origin=cspace.cspace.classes.Attribute.Origin(
                (0, -0.22, 0.25, 0, 0, 0),
            ),
            motion=cspace.cspace.classes.Attribute.Motion("fixed", (1, 0, 0), 0, 0),
        ),
        "right_base_joint": cspace.cspace.classes.Joint(
            name="right_base_joint",
            child="right_base",
            parent="right_leg",
            origin=cspace.cspace.classes.Attribute.Origin(
                (0, 0, -0.6, 0, 0, 0),
            ),
            motion=cspace.cspace.classes.Attribute.Motion("fixed", (1, 0, 0), 0, 0),
        ),
        "right_front_wheel_joint": cspace.cspace.classes.Joint(
            name="right_front_wheel_joint",
            child="right_front_wheel",
            parent="right_base",
            origin=cspace.cspace.classes.Attribute.Origin(
                (0.133333333333, 0, -0.085, 0, 0, 0),
            ),
            motion=cspace.cspace.classes.Attribute.Motion(
                "continuous", (0, 1, 0), 0, 0
            ),
        ),
        "right_back_wheel_joint": cspace.cspace.classes.Joint(
            name="right_back_wheel_joint",
            child="right_back_wheel",
            parent="right_base",
            origin=cspace.cspace.classes.Attribute.Origin(
                (-0.133333333333, 0, -0.085, 0, 0, 0),
            ),
            motion=cspace.cspace.classes.Attribute.Motion(
                "continuous", (0, 1, 0), 0, 0
            ),
        ),
        "base_to_left_leg": cspace.cspace.classes.Joint(
            name="base_to_left_leg",
            child="left_leg",
            parent="base_link",
            origin=cspace.cspace.classes.Attribute.Origin(
                (0, 0.22, 0.25, 0, 0, 0),
            ),
            motion=cspace.cspace.classes.Attribute.Motion("fixed", (1, 0, 0), 0, 0),
        ),
        "left_base_joint": cspace.cspace.classes.Joint(
            name="left_base_joint",
            child="left_base",
            parent="left_leg",
            origin=cspace.cspace.classes.Attribute.Origin(
                (0, 0, -0.6, 0, 0, 0),
            ),
            motion=cspace.cspace.classes.Attribute.Motion("fixed", (1, 0, 0), 0, 0),
        ),
        "left_front_wheel_joint": cspace.cspace.classes.Joint(
            name="left_front_wheel_joint",
            child="left_front_wheel",
            parent="left_base",
            origin=cspace.cspace.classes.Attribute.Origin(
                (0.133333333333, 0, -0.085, 0, 0, 0),
            ),
            motion=cspace.cspace.classes.Attribute.Motion(
                "continuous", (0, 1, 0), 0, 0
            ),
        ),
        "left_back_wheel_joint": cspace.cspace.classes.Joint(
            name="left_back_wheel_joint",
            child="left_back_wheel",
            parent="left_base",
            origin=cspace.cspace.classes.Attribute.Origin(
                (-0.133333333333, 0, -0.085, 0, 0, 0),
            ),
            motion=cspace.cspace.classes.Attribute.Motion(
                "continuous", (0, 1, 0), 0, 0
            ),
        ),
        "gripper_extension": cspace.cspace.classes.Joint(
            name="gripper_extension",
            child="gripper_pole",
            parent="base_link",
            origin=cspace.cspace.classes.Attribute.Origin(
                (0.19, 0, 0.2, 0, 0, 0),
            ),
            motion=cspace.cspace.classes.Attribute.Motion(
                "prismatic", (1, 0, 0), -0.38, 0.0
            ),
        ),
        "left_gripper_joint": cspace.cspace.classes.Joint(
            name="left_gripper_joint",
            child="left_gripper",
            parent="gripper_pole",
            origin=cspace.cspace.classes.Attribute.Origin(
                (0.2, 0.01, 0, 0, 0, 0),
            ),
            motion=cspace.cspace.classes.Attribute.Motion(
                "revolute", (0, 0, 1), 0.0, 0.548
            ),
        ),
        "left_tip_joint": cspace.cspace.classes.Joint(
            name="left_tip_joint",
            child="left_tip",
            parent="left_gripper",
            origin=cspace.cspace.classes.Attribute.Origin(
                (0, 0, 0, 0, 0, 0),
            ),
            motion=cspace.cspace.classes.Attribute.Motion("fixed", (1, 0, 0), 0, 0),
        ),
        "right_gripper_joint": cspace.cspace.classes.Joint(
            name="right_gripper_joint",
            child="right_gripper",
            parent="gripper_pole",
            origin=cspace.cspace.classes.Attribute.Origin(
                (0.2, -0.01, 0, 0, 0, 0),
            ),
            motion=cspace.cspace.classes.Attribute.Motion(
                "revolute", (0, 0, -1), 0.0, 0.548
            ),
        ),
        "right_tip_joint": cspace.cspace.classes.Joint(
            name="right_tip_joint",
            child="right_tip",
            parent="right_gripper",
            origin=cspace.cspace.classes.Attribute.Origin(
                (0, 0, 0, 0, 0, 0),
            ),
            motion=cspace.cspace.classes.Attribute.Motion("fixed", (1, 0, 0), 0, 0),
        ),
        "head_swivel": cspace.cspace.classes.Joint(
            name="head_swivel",
            child="head",
            parent="base_link",
            origin=cspace.cspace.classes.Attribute.Origin(
                (0, 0, 0.3, 0, 0, 0),
            ),
            motion=cspace.cspace.classes.Attribute.Motion(
                "continuous", (0, 0, 1), 0, 0
            ),
        ),
        "tobox": cspace.cspace.classes.Joint(
            name="tobox",
            child="box",
            parent="head",
            origin=cspace.cspace.classes.Attribute.Origin(
                (0.1814, 0, 0.1414, 0, 0, 0),
            ),
            motion=cspace.cspace.classes.Attribute.Motion("fixed", (1, 0, 0), 0, 0),
        ),
    }
    for name, joint in joints.items():
        assert spec.joint(name) == joint
    assert len(joints) == len(spec.joint)

    links = {
        "base_link",
        "right_leg",
        "right_base",
        "right_front_wheel",
        "right_back_wheel",
        "left_leg",
        "left_base",
        "left_front_wheel",
        "left_back_wheel",
        "gripper_pole",
        "left_gripper",
        "left_tip",
        "right_gripper",
        "right_tip",
        "head",
        "box",
    }
    assert links == {item[-1] for item in spec.chain}

    chains = [
        tuple(["base_link", "right_leg", "right_base", "right_back_wheel"]),
        tuple(["base_link", "gripper_pole", "left_gripper", "left_tip"]),
        tuple(["base_link", "right_leg", "right_base", "right_front_wheel"]),
        tuple(["base_link", "left_leg"]),
        tuple(["base_link", "right_leg"]),
        tuple(["base_link", "gripper_pole", "right_gripper"]),
        tuple(["base_link", "left_leg", "left_base", "left_front_wheel"]),
        tuple(["base_link", "head"]),
        tuple(["base_link", "gripper_pole", "left_gripper"]),
        tuple(["base_link", "head", "box"]),
        tuple(["base_link", "right_leg", "right_base"]),
        tuple(["base_link"]),
        tuple(["base_link", "left_leg", "left_base"]),
        tuple(["base_link", "gripper_pole", "right_gripper", "right_tip"]),
        tuple(["base_link", "left_leg", "left_base", "left_back_wheel"]),
        tuple(["base_link", "gripper_pole"]),
    ]
    assert sorted(chains) == sorted(spec.chain)

    def lookup(joints, source, target):
        for name, joint in joints.items():
            if joint.parent == target and joint.child == source:
                return (name, True)
            elif joint.parent == source and joint.child == target:
                return (name, False)

    for source in links:
        for target in links:
            source_chain = tuple(
                reversed(next(filter(lambda item: item[-1] == source, chains)))
            )
            target_chain = next(filter(lambda item: item[-1] == target, chains))[1:]
            route = source_chain + target_chain
            forward_route = route[: route.index(target) + 1]
            reverse_route = route[
                len(route) - 1 - list(reversed(route)).index(source) :
            ]
            final_route = (
                forward_route
                if len(forward_route) <= len(reverse_route)
                else reverse_route
            )
            joint_route = [
                lookup(joints, final_route[i - 1], final_route[i])
                for i in range(1, len(final_route))
            ]
            logging.getLogger(__name__).info(
                "".join(
                    [
                        f"\n----------------------------------------",
                        f"\n[{source} => {target}]",
                        f"\nROUTE: {route}",
                        f"\nFINAL: {final_route}",
                        f"\nJOINT: {[(name, joints[name].parent, joints[name].child, 'forward' if forward else 'inverse') for name, forward in joint_route]}",
                        f"\n----------------------------------------",
                        f"\n",
                    ]
                )
            )
            assert joint_route == spec.route(source, target)


def test_spec(device, urdf_file_panda):
    with pytest.raises(AssertionError, match="TODO: mimic"):
        spec = cspace.cspace.classes.Spec(
            description=pathlib.Path(urdf_file_panda).read_text()
        )
