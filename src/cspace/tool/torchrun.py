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
        default="train",
        choices=["data", "train"],
    )
    mode = parser.parse_known_args()[0].mode
    if mode == "data":
        parser.add_argument("--data", dest="data", type=str, required=True)
        parser.add_argument("--model", dest="model", type=str, required=True)
        parser.add_argument("--urdf", dest="urdf", type=str, required=True)
        parser.add_argument("--link", dest="link", type=str, nargs="+", default=[])
        parser.add_argument("--seed", dest="seed", type=int, default=0)
        parser.add_argument("--noise", dest="noise", type=int, default=None)
        parser.add_argument("--total", dest="total", type=int, default=1024)
        parser.add_argument("--bucket", dest="bucket", type=int, default=None)
    elif mode == "train":
        parser.add_argument("--data", dest="data", type=str, required=True)
        parser.add_argument("--model", dest="model", type=str, required=True)
        parser.add_argument("--batch", dest="batch", type=int, default=16)
        parser.add_argument("--epoch", dest="epoch", type=int, default=5)
        parser.add_argument("--save", dest="save", type=str, default=None)
    else:
        assert False, "{}".format(mode)

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)8s] %(message)s (%(filename)s:%(lineno)s)",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    accelerator = accelerate.Accelerator()
    logger = accelerate.logging.get_logger(__name__)
    logger.setLevel(logging.INFO)

    logger.info(f"Args: {args}")

    if mode == "data":
        kinematics = cspace.transformers.InverseKinematics(
            pathlib.Path(args.urdf).read_text(),
            *args.link,
            model="gpt2",
            bucket=args.bucket,
        )
        dataset = cspace.transformers.InverseDataset(
            *kinematics.rand(
                logger=logger,
                total=args.total,
                noise=args.noise,
                seed=args.seed,
            )
        )
        accelerator.save(dataset, args.data)
        accelerator.save(kinematics, args.model)

    else:
        dataset = torch.load(args.data)
        kinematics = torch.load(args.model)

        kinematics.train(
            logger=logger,
            dataset=dataset,
            accelerator=accelerator,
            batch=args.batch,
            epoch=args.epoch,
            save=args.save,
        )


if __name__ == "__main__":
    main()
