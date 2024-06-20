import cspace.transformers
import tempfile
import argparse
import pathlib
import logging
import sys
import torch
import torch.distributed.run


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
        "--urdf",
        dest="urdf",
        type=str,
        required=True,
    )
    parser.add_argument("--load", dest="load", type=str, default=None)
    parser.add_argument("--save", dest="save", type=str, default=None)
    parser.add_argument("--link", dest="link", type=str, nargs="+", required=True)
    parser.add_argument("--seed", dest="seed", type=int, default=0)
    parser.add_argument("--total", dest="total", type=int, default=1024)
    parser.add_argument("--batch", dest="batch", type=int, default=16)
    parser.add_argument("--noise", dest="noise", type=int, default=None)
    parser.add_argument("--epoch", dest="epoch", type=int, default=5)

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)8s] %(message)s (%(filename)s:%(lineno)s)",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger(__name__).info(f"Args: {args}")

    kinematics = cspace.transformers.Kinematics(
        pathlib.Path(args.urdf).read_text(), *args.link, model="gpt2"
    )
    kinematics.model = (
        accelerate.load_checkpoint_and_dispatch(kinematics.model, checkpoint=args.load)
        if args.load
        else kinematics.model
    )

    kinematics.train(
        seed=args.seed,
        total=args.total,
        batch=args.batch,
        noise=args.noise,
        epoch=args.epoch,
        save=args.save,
    )


if __name__ == "__main__":
    main()
