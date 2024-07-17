import cspace.transformers
import argparse
import logging


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", dest="load", type=str, required=True)
    parser.add_argument("--joint", dest="joint", type=str, nargs="+", required=True)

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)8s] %(message)s (%(filename)s:%(lineno)s)",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger(__name__).info(f"Args: {args}")

    kinematics = cspace.transformers.InverseKinematics.load(args.load)

    joint, position = zip(*tuple(e.split(":", maxsplit=1) for e in args.joint))
    joint, position = tuple(joint), tuple(float(e) for e in position)
    state = cspace.torch.classes.JointStateCollection(joint, position)

    assert joint == kinematics.joint

    pose = kinematics.forward(state)

    inverse = kinematics.inverse(pose)
    logging.getLogger(__name__).info(
        (
            "\n" + "\n[Inverse Kinematics]" + "\n" + "\nPred: {}" + "\nTrue: {}" + "\n"
        ).format(
            list(
                (name, inverse.position(kinematics.spec, name)) for name in inverse.name
            ),
            list((name, state.position(kinematics.spec, name)) for name in state.name),
        )
    )


if __name__ == "__main__":
    main()
