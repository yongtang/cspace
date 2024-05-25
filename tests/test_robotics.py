import cspace.robotics

import pathlib


def test_description(device, urdf_file):
    description = cspace.robotics.Description(
        description=pathlib.Path(urdf_file).read_text()
    )

    joints = {
        "base_to_right_leg": cspace.robotics.Fixed(
            name="base_to_right_leg",
            child="right_leg",
            parent="base_link",
            origin=(0, -0.22, 0.25, 0, 0, 0),
        ),
        "right_base_joint": cspace.robotics.Fixed(
            name="right_base_joint",
            child="right_base",
            parent="right_leg",
            origin=(0, 0, -0.6, 0, 0, 0),
        ),
        "right_front_wheel_joint": cspace.robotics.Continuous(
            name="right_front_wheel_joint",
            child="right_front_wheel",
            parent="right_base",
            origin=(0.133333333333, 0, -0.085, 0, 0, 0),
            axis=(0, 1, 0),
        ),
        "right_back_wheel_joint": cspace.robotics.Continuous(
            name="right_back_wheel_joint",
            child="right_back_wheel",
            parent="right_base",
            origin=(-0.133333333333, 0, -0.085, 0, 0, 0),
            axis=(0, 1, 0),
        ),
        "base_to_left_leg": cspace.robotics.Fixed(
            name="base_to_left_leg",
            child="left_leg",
            parent="base_link",
            origin=(0, 0.22, 0.25, 0, 0, 0),
        ),
        "left_base_joint": cspace.robotics.Fixed(
            name="left_base_joint",
            child="left_base",
            parent="left_leg",
            origin=(0, 0, -0.6, 0, 0, 0),
        ),
        "left_front_wheel_joint": cspace.robotics.Continuous(
            name="left_front_wheel_joint",
            child="left_front_wheel",
            parent="left_base",
            origin=(0.133333333333, 0, -0.085, 0, 0, 0),
            axis=(0, 1, 0),
        ),
        "left_back_wheel_joint": cspace.robotics.Continuous(
            name="left_back_wheel_joint",
            child="left_back_wheel",
            parent="left_base",
            origin=(-0.133333333333, 0, -0.085, 0, 0, 0),
            axis=(0, 1, 0),
        ),
        "gripper_extension": cspace.robotics.Prismatic(
            name="gripper_extension",
            child="gripper_pole",
            parent="base_link",
            origin=(0.19, 0, 0.2, 0, 0, 0),
            axis=(1, 0, 0),
            limit=cspace.robotics.Joint.Limit(
                lower=-0.38, upper=0, effort=1000.0, velocity=0.5
            ),
        ),
        "left_gripper_joint": cspace.robotics.Revolute(
            name="left_gripper_joint",
            child="left_gripper",
            parent="gripper_pole",
            origin=(0.2, 0.01, 0, 0, 0, 0),
            axis=(0, 0, 1),
            limit=cspace.robotics.Joint.Limit(
                lower=0.0, upper=0.548, effort=1000.0, velocity=0.5
            ),
        ),
        "left_tip_joint": cspace.robotics.Fixed(
            name="left_tip_joint",
            child="left_tip",
            parent="left_gripper",
            origin=(0, 0, 0, 0, 0, 0),
        ),
        "right_gripper_joint": cspace.robotics.Revolute(
            name="right_gripper_joint",
            child="right_gripper",
            parent="gripper_pole",
            origin=(0.2, -0.01, 0, 0, 0, 0),
            axis=(0, 0, -1),
            limit=cspace.robotics.Joint.Limit(
                lower=0.0, upper=0.548, effort=1000.0, velocity=0.5
            ),
        ),
        "right_tip_joint": cspace.robotics.Fixed(
            name="right_tip_joint",
            child="right_tip",
            parent="right_gripper",
            origin=(0, 0, 0, 0, 0, 0),
        ),
        "head_swivel": cspace.robotics.Continuous(
            name="head_swivel",
            child="head",
            parent="base_link",
            origin=(0, 0, 0.3, 0, 0, 0),
            axis=(0, 0, 1),
        ),
        "tobox": cspace.robotics.Fixed(
            name="tobox",
            child="box",
            parent="head",
            origin=(0.1814, 0, 0.1414, 0, 0, 0),
        ),
    }
    for name, joint in joints.items():
        assert description.joint(name) == joint
    assert len(joints) == len(description.joint)

    links = set(
        [
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
        ]
    )
    assert links == description.link

    chains = [
        ["base_link", "right_leg", "right_base", "right_back_wheel"],
        ["base_link", "gripper_pole", "left_gripper", "left_tip"],
        ["base_link", "right_leg", "right_base", "right_front_wheel"],
        ["base_link", "left_leg"],
        ["base_link", "right_leg"],
        ["base_link", "gripper_pole", "right_gripper"],
        ["base_link", "left_leg", "left_base", "left_front_wheel"],
        ["base_link", "head"],
        ["base_link", "gripper_pole", "left_gripper"],
        ["base_link", "head", "box"],
        ["base_link", "right_leg", "right_base"],
        ["base_link"],
        ["base_link", "left_leg", "left_base"],
        ["base_link", "gripper_pole", "right_gripper", "right_tip"],
        ["base_link", "left_leg", "left_base", "left_back_wheel"],
        ["base_link", "gripper_pole"],
    ]
    assert sorted(chains) == sorted(description.chain)
