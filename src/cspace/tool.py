import cspace.transformers
import argparse
import pathlib
import logging
import torch


def main():
    command = argparse.ArgumentParser()
    command.add_argument(
        "mode",
        type=str,
        choices=["inference", "train"],
    )

    choice = command.parse_known_args()[0].mode

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        type=str,
        choices=["inference", "train"],
    )
    parser.add_argument(
        "--urdf",
        dest="urdf",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--model",
        dest="model",
        type=str,
        required=True,
    )
    parser.add_argument("--link", dest="link", type=str, nargs="+", required=True)
    if choice == "train":
        parser.add_argument("--seed", dest="seed", type=int, default=0)
        parser.add_argument("--total", dest="total", type=int, default=1024)
        parser.add_argument("--batch", dest="batch", type=int, default=16)
        parser.add_argument("--noise", dest="noise", type=int, default=None)
        parser.add_argument("--epoch", dest="epoch", type=int, default=5)
        parser.add_argument("--device", dest="device", type=str, default="cpu")
    else:
        parser.add_argument("--joint", dest="joint", type=str, nargs="+", required=True)

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logging.getLogger(__name__).info(f"Args: {args}")

    kinematics = cspace.transformers.Kinematics(
        pathlib.Path(args.urdf).read_text(), *args.link, model="gpt2"
    )

    if choice == "train":
        kinematics.train(
            seed=args.seed,
            total=args.total,
            batch=args.batch,
            noise=args.noise,
            epoch=args.epoch,
            device=(torch.device(args.device) if args.device else None),
        )

        torch.save(kinematics.model.state_dict(), args.model)

        logging.getLogger(__name__).info(f"Model save {args.model}")
    else:

        kinematics.model.load_state_dict(torch.load(args.model))
        logging.getLogger(__name__).info(f"Model load {args.model}")

        joint, position = zip(*tuple(e.split(":", maxsplit=1) for e in args.joint))
        joint, position = tuple(joint), tuple(float(e) for e in position)
        state = cspace.torch.classes.JointStateCollection(joint, position)

        pose = kinematics.forward(state)

        inverse = kinematics.inverse(pose)
        logging.getLogger(__name__).info(
            (
                "\n"
                + "\n[Inverse Kinematics]"
                + "\n"
                + "\nPred: {}"
                + "\nTrue: {}"
                + "\n"
            ).format(
                list(
                    (name, inverse.position(kinematics.spec, name))
                    for name in inverse.name
                ),
                list(
                    (name, state.position(kinematics.spec, name)) for name in state.name
                ),
            )
        )


if __name__ == "__main__":
    main()
