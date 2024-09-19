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
    if "--torchrun" in sys.argv:
        index = sys.argv.index("--torchrun")
        with tempfile.NamedTemporaryFile(
            suffix=str(pathlib.Path(__file__)).replace("/", "_._")
        ) as f:
            pathlib.Path(f.name).write_text(pathlib.Path(__file__).read_text())
            sys.argv = (
                sys.argv[:1] + sys.argv[index + 1 :] + [f.name] + sys.argv[1:index]
            )
            sys.exit(torch.distributed.run.main())

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        dest="mode",
        type=str,
        default="check",
        choices=["check", "train"],
    )
    parser.add_argument(
        "--func",
        dest="func",
        type=str,
        default="inverse",
        choices=["inverse", "perception"],
    )
    mode = parser.parse_known_args()[0].mode
    func = parser.parse_known_args()[0].func

    if mode == "check":
        if func == "inverse":
            parser.add_argument("--load", dest="load", type=str, required=True)
            parser.add_argument(
                "--joint", dest="joint", type=str, nargs="+", required=True
            )
            parser.add_argument(
                "--start", dest="start", type=str, nargs="+", default=[]
            )
            parser.add_argument("--repeat", dest="repeat", type=int, default=None)
            parser.add_argument("--device", dest="device", type=str, default="cpu")
        else:
            parser.add_argument("--load", dest="load", type=str, required=True)
            parser.add_argument("--image", dest="image", type=str, required=True)
            parser.add_argument("--device", dest="device", type=str, default="cpu")
    else:
        if func == "inverse":
            parser.add_argument("--save", dest="save", type=str, required=True)
            parser.add_argument("--load", dest="load", type=str, default=None)

            parser.add_argument("--total", dest="total", type=int, default=None)

            parser.add_argument("--batch", dest="batch", type=int, default=None)
            parser.add_argument("--epoch-start", dest="start", type=int, default=None)
            parser.add_argument("--epoch-limit", dest="limit", type=int, default=None)
            load = parser.parse_known_args()[0].load

            if not load:
                parser.add_argument("--urdf", dest="urdf", type=str, required=True)
                parser.add_argument(
                    "--link", dest="link", type=str, nargs="+", default=[]
                )
                parser.add_argument("--bucket", dest="bucket", type=int, default=None)
                parser.add_argument("--length", dest="length", type=int, default=None)
        else:
            parser.add_argument("--save", dest="save", type=str, required=True)
            parser.add_argument("--load", dest="load", type=str, default=None)

            parser.add_argument("--total", dest="total", type=total, default=None)

            parser.add_argument("--batch", dest="batch", type=int, default=None)
            parser.add_argument("--epoch-start", dest="start", type=int, default=None)
            parser.add_argument("--epoch-limit", dest="limit", type=int, default=None)
            load = parser.parse_known_args()[0].load

            if not load:
                parser.add_argument("--urdf", dest="urdf", type=str, required=True)
                parser.add_argument("--bucket", dest="bucket", type=int, default=None)
                parser.add_argument("--length", dest="length", type=int, default=None)
            else:
                parser.add_argument("--device", dest="device", type=str, default="cpu")

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
            kinematics = torch.load(args.load, map_location=torch.device(args.device))

            if func == "inverse":
                joint, position = zip(
                    *tuple(e.split(":", maxsplit=1) for e in args.joint)
                )
                joint, position = tuple(joint), torch.tensor(
                    tuple(float(e) for e in position), device=torch.device(args.device)
                )

                state = cspace.torch.classes.JointStateCollection(
                    kinematics.spec, joint, position
                )
                assert joint == kinematics.joint, "{} vs. {}".format(
                    joint, kinematics.joint
                )
                pose = kinematics.forward(state)

                zero = cspace.torch.classes.JointStateCollection.apply(
                    kinematics.spec,
                    kinematics.joint,
                    torch.zeros(
                        pose.batch + tuple([len(kinematics.joint)]),
                        device=torch.device(args.device),
                    ),
                    min=0.0,
                    max=1.0,
                )
                mark = kinematics.forward(zero)

                if args.start:
                    joint, position = zip(
                        *tuple(e.split(":", maxsplit=1) for e in args.start)
                    )
                    joint, position = tuple(joint), torch.tensor(
                        tuple(float(e) for e in position),
                        device=torch.device(args.device),
                    )

                    start = cspace.torch.classes.JointStateCollection(
                        kinematics.spec, joint, position
                    )
                    assert joint == kinematics.joint, "{} vs. {}".format(
                        joint, kinematics.joint
                    )
                    node = kinematics.forward(start)
                else:
                    start = zero
                    node = mark

                inverse = kinematics.inverse(pose, start, repeat=args.repeat)
                pred = kinematics.forward(inverse)

                logger.info(
                    (
                        "\n"
                        + "[Inverse Kinematics]\n"
                        + "\n"
                        + "Limit:{}\n"
                        + "\n"
                        + "--------------------\n"
                        + "\n"
                        + "Zero: {}\n"
                        + "\n"
                        + "Pose: [position]    {}\n"
                        + "      [orientation] {}\n"
                        + "\n"
                        + "--------------------\n"
                        + "\n"
                        + "Node: {}\n"
                        + "\n"
                        + "Pose: [position]    {}\n"
                        + "      [orientation] {}\n"
                        + "\n"
                        + "--------------------\n"
                        + "\n"
                        + "True: {}\n"
                        + "\n"
                        + "Pose: [position]    {}\n"
                        + "      [orientation] {}\n"
                        + "\n"
                        + "--------------------\n"
                        + "\n"
                        + "Pred: {}\n"
                        + "\n"
                        + "Pose: [position]    {}\n"
                        + "      [orientation] {}\n"
                        + "\n"
                    ).format(
                        list(
                            (name, kinematics.spec.joint(name).motion.limit)
                            for name in zero.name
                        ),
                        list(
                            (
                                name,
                                zero.position(kinematics.spec, name).data.cpu().item(),
                            )
                            for name in zero.name
                        ),
                        list(
                            (name, mark.position(name).data.cpu().tolist())
                            for name in mark.name
                        ),
                        list(
                            (name, mark.orientation(name).data.cpu().tolist())
                            for name in mark.name
                        ),
                        list(
                            (
                                name,
                                start.position(kinematics.spec, name).data.cpu().item(),
                            )
                            for name in start.name
                        ),
                        list(
                            (name, node.position(name).data.cpu().tolist())
                            for name in node.name
                        ),
                        list(
                            (name, node.orientation(name).data.cpu().tolist())
                            for name in node.name
                        ),
                        list(
                            (
                                name,
                                state.position(kinematics.spec, name).data.cpu().item(),
                            )
                            for name in state.name
                        ),
                        list(
                            (name, pose.position(name).data.cpu().tolist())
                            for name in pose.name
                        ),
                        list(
                            (name, pose.orientation(name).data.cpu().tolist())
                            for name in pose.name
                        ),
                        list(
                            (
                                name,
                                inverse.position(kinematics.spec, name)
                                .data.cpu()
                                .item(),
                            )
                            for name in inverse.name
                        ),
                        list(
                            (name, pred.position(name).data.cpu().numpy().tolist())
                            for name in pred.name
                        ),
                        list(
                            (name, pred.orientation(name).data.cpu().numpy().tolist())
                            for name in pred.name
                        ),
                    )
                )
            else:
                observation = kinematics.image(
                    args.image, device=torch.device(args.device)
                )
                perception = kinematics.perception(observation)

                pred = kinematics.forward(perception)

                zero = cspace.torch.classes.JointStateCollection.apply(
                    kinematics.spec,
                    kinematics.joint,
                    torch.zeros(
                        pred.batch + tuple([len(kinematics.joint)]),
                        device=torch.device(args.device),
                    ),
                    min=0.0,
                    max=1.0,
                )
                mark = kinematics.forward(zero)

                logger.info(
                    (
                        "\n"
                        + "[Perception Kinematics]\n"
                        + "\n"
                        + "Limit:{}\n"
                        + "\n"
                        + "--------------------\n"
                        + "\n"
                        + "Zero: {}\n"
                        + "\n"
                        + "Pose: [position]    {}\n"
                        + "      [orientation] {}\n"
                        + "\n"
                        + "--------------------\n"
                        + "\n"
                        + "Pred: {}\n"
                        + "\n"
                        + "Pose: [position]    {}\n"
                        + "      [orientation] {}\n"
                        + "\n"
                    ).format(
                        list(
                            (name, kinematics.spec.joint(name).motion.limit)
                            for name in zero.name
                        ),
                        list(
                            (
                                name,
                                zero.position(kinematics.spec, name).data.cpu().item(),
                            )
                            for name in zero.name
                        ),
                        list(
                            (name, mark.position(name).data.cpu().tolist())
                            for name in mark.name
                        ),
                        list(
                            (name, mark.orientation(name).data.cpu().tolist())
                            for name in mark.name
                        ),
                        list(
                            (
                                name,
                                perception.position(kinematics.spec, name)
                                .data.cpu()
                                .item(),
                            )
                            for name in perception.name
                        ),
                        list(
                            (name, pred.position(name).data.cpu().numpy().tolist())
                            for name in pred.name
                        ),
                        list(
                            (name, pred.orientation(name).data.cpu().numpy().tolist())
                            for name in pred.name
                        ),
                    )
                )

    else:
        if func == "inverse":
            kinematics = (
                torch.load(pathlib.Path(args.load).joinpath("kinematics.pth"))
                if args.load
                else cspace.transformers.InverseKinematics(
                    pathlib.Path(args.urdf).read_text(),
                    *args.link,
                    model="gpt2",
                    bucket=args.bucket,
                    length=args.length,
                )
            )
            kinematics.train(
                logger=logger,
                accelerator=accelerator,
                total=args.total,
                load=args.load,
                save=args.save,
                batch=args.batch,
                start=args.start,
                limit=args.limit,
            )
        else:
            with pathlib.Path(total).open() as f:
                total = list(entry.strip() for entry in f)

            kinematics = (
                torch.load(args.load)
                if args.load
                else cspace.transformers.PerceptionKinematics(
                    pathlib.Path(args.urdf).read_text(),
                    model="gpt2",
                    vision="google/vit-base-patch16-224",
                    bucket=args.bucket,
                    length=args.length,
                )
            )
            kinematics.train(
                logger=logger,
                accelerator=accelerator,
                total=total,
                load=args.load,
                save=args.save,
                batch=args.batch,
                start=args.start,
                limit=args.limit,
            )


if __name__ == "__main__":
    main()
