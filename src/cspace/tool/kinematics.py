import cspace.transformers
import argparse
import pathlib
import logging
import accelerate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--urdf",
        dest="urdf",
        type=str,
        required=True,
    )
    parser.add_argument("--load", dest="load", type=str, required=True)
    parser.add_argument("--link", dest="link", type=str, nargs="+", required=True)
    parser.add_argument("--joint", dest="joint", type=str, nargs="+", required=True)

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logging.getLogger(__name__).info(f"Args: {args}")

    kinematics = cspace.transformers.Kinematics(
        pathlib.Path(args.urdf).read_text(), *args.link, model="gpt2"
    )

    joint, position = zip(*tuple(e.split(":", maxsplit=1) for e in args.joint))
    joint, position = tuple(joint), tuple(float(e) for e in position)
    state = cspace.torch.classes.JointStateCollection(joint, position)

    # initialize parameters
    kinematics.inverse(kinematics.forward(state))

    kinematics.model = accelerate.load_checkpoint_and_dispatch(
        kinematics.model, checkpoint=args.load
    )

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
