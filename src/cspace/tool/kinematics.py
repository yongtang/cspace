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

            parser.add_argument("--noise", dest="noise", type=int, default=None)

            parser.add_argument("--batch", dest="batch", type=int, default=None)
            parser.add_argument("--lr", dest="lr", type=float, default=None)
            load = parser.parse_known_args()[0].load

            if not load:
                parser.add_argument("--urdf", dest="urdf", type=str, required=True)
                parser.add_argument(
                    "--link", dest="link", type=str, nargs="+", default=[]
                )
                parser.add_argument("--bucket", dest="bucket", type=int, default=None)
                parser.add_argument("--length", dest="length", type=int, default=None)
            else:
                parser.add_argument("--device", dest="device", type=str, default="cpu")
        else:
            parser.add_argument("--save", dest="save", type=str, required=True)
            parser.add_argument("--load", dest="load", type=str, default=None)

            parser.add_argument("--image", dest="image", type=str, default=None)
            parser.add_argument("--label", dest="label", type=str, default=None)

            parser.add_argument("--batch", dest="batch", type=int, default=None)
            parser.add_argument("--lr", dest="lr", type=float, default=None)
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
                joint, position = tuple(joint), tuple(float(e) for e in position)

                state = cspace.torch.classes.JointStateCollection(joint, position)
                assert joint == kinematics.joint, "{} vs. {}".format(
                    joint, kinematics.joint
                )
                pose = kinematics.forward(state)

                zero = cspace.torch.classes.JointStateCollection.apply(
                    kinematics.spec,
                    kinematics.joint,
                    torch.zeros(pose.batch + tuple([len(kinematics.joint)])),
                    min=-1.0,
                    max=1.0,
                )
                mark = kinematics.forward(zero)

                inverse = kinematics.inverse(pose, zero, repeat=args.repeat)
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
                observation = kinematics.image(image)
                perception = kinematics.perception(observation)

                pred = kinematics.forward(perception)

                zero = cspace.torch.classes.JointStateCollection.apply(
                    kinematics.spec,
                    kinematics.joint,
                    torch.zeros(pose.batch + tuple([len(kinematics.joint)])),
                    min=-1.0,
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
                torch.load(args.load, map_location=torch.device(args.device))
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
                save=args.save,
                batch=args.batch,
                total=args.total,
                lr=args.lr,
            )
        else:
            kinematics = (
                torch.load(args.load, map_location=torch.device(args.device))
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
                image=args.image,
                label=args.label,
                save=args.save,
                batch=args.batch,
                lr=args.lr,
            )


if __name__ == "__main__":
    main()
