import cspace.transformers
import tempfile
import argparse
import pathlib
import logging
import sys
import torch
import torch.distributed.run
import accelerate


def main():
    if "--" in sys.argv:
        index = sys.argv.index("--")
        with tempfile.NamedTemporaryFile(
            suffix=str(pathlib.Path(__file__)).replace("/", "_._")
        ) as f:
            pathlib.Path(f.name).write_text(pathlib.Path(__file__).read_text())
            sys.argv = sys.argv[:index] + [f.name] + sys.argv[index + 1 :]

            sys.exit(torch.distributed.run.main())

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        dest="mode",
        type=str,
        default="check",
        choices=["check", "build", "train"],
    )
    mode = parser.parse_known_args()[0].mode

    if mode == "check":
        parser.add_argument("--load", dest="load", type=str, required=True)
        parser.add_argument("--joint", dest="joint", type=str, nargs="+", required=True)
    elif mode == "build":
        parser.add_argument("--data", dest="data", type=str, required=True)
        parser.add_argument("--model", dest="model", type=str, required=True)
        parser.add_argument("--urdf", dest="urdf", type=str, required=True)
        parser.add_argument("--link", dest="link", type=str, nargs="+", default=[])
        parser.add_argument("--seed", dest="seed", type=int, default=0)
        parser.add_argument("--noise", dest="noise", type=int, default=None)
        parser.add_argument("--total", dest="total", type=int, default=1024)
        parser.add_argument("--bucket", dest="bucket", type=int, default=None)
    else:
        parser.add_argument("--data", dest="data", type=str, required=True)
        parser.add_argument("--model", dest="model", type=str, required=True)
        parser.add_argument("--batch", dest="batch", type=int, default=16)
        parser.add_argument("--epoch", dest="epoch", type=int, default=5)
        parser.add_argument("--save", dest="save", type=str, default=None)

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)8s] %(message)s (%(filename)s:%(lineno)s)",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    accelerator = accelerate.Accelerator()
    logger = accelerate.logging.get_logger(__name__)

    logger.info(f"Args: {args}")

    if mode == "check":
        with accelerator.main_process_first():
            kinematics = torch.load(args.load)

            joint, position = zip(*tuple(e.split(":", maxsplit=1) for e in args.joint))
            joint, position = tuple(joint), tuple(float(e) for e in position)
            state = cspace.torch.classes.JointStateCollection(joint, position)

            assert joint == kinematics.joint, "{} vs. {}".format(
                joint, kinematics.joint
            )

            pose = kinematics.forward(state)

            inverse = kinematics.inverse(pose)
        logger.info(
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
    elif mode == "build":
        with accelerator.main_process_first():
            kinematics = cspace.transformers.InverseKinematics(
                pathlib.Path(args.urdf).read_text(),
                *args.link,
                model="gpt2",
                bucket=args.bucket,
            )
            kinematics.rand(
                logger=logger,
                save=args.data,
                total=args.total,
                noise=args.noise,
                seed=args.seed,
            )
            torch.save(kinematics, args.model)
    else:
        kinematics = torch.load(args.model)

        with accelerator.main_process_first():
            dataset = cspace.transformers.InverseDataset(data=args.data, logger=logger)

        kinematics.train(
            logger=logger,
            accelerator=accelerator,
            dataset=dataset,
            batch=args.batch,
            epoch=args.epoch,
            save=args.save,
        )


if __name__ == "__main__":
    main()
