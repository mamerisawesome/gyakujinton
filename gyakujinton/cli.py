import argparse
import sys


def cli():
    commands = {
        "draw_on_image": shape_on_image,
        "distort": skew_image,
        "batch_distort": batch_skew,
    }

    options = {
        "prog": "gyakujinton",
        "usage": '%(prog)s [options]',
        "description": "OpenCV wrapper to handle shapes and images."
    }

    for c in commands.keys():
        if c in sys.argv:
            parser = argparse.ArgumentParser(
                add_help=False,
                **options
            )
            commands[c](parser, sys.argv[2:])
            return

    parser = argparse.ArgumentParser(**options)

    args = parser.parse_args()
    print(args)


def shape_on_image(parent_parser, arguments):
    from .functions.draw import draw_on_image

    parser = argparse.ArgumentParser(
        prog="draw_on_image",
        parents=[parent_parser]
    )

    parser.add_argument(
        "image_path",
        help="file path of image to be drawn on"
    )

    parser.add_argument(
        "-o",
        "--output_path",
        help="output path of image with the modifications"
    )

    parser.add_argument(
        "-p",
        "--points",
        nargs="+",
        action="append",
        required=True,
        help="x,y points on a 2D plane; e.g. 1,2 3,4, 5,6"
    )

    args = parser.parse_args(arguments)
    draw_on_image(
        image_path=args.image_path,
        output_path=args.output_path,
        points=[
            [int(c) for c in point.split(",")] for point in args.points[-1]
        ]
    )

    return 0


def skew_image(parent_parser, arguments):
    from .functions.skew import skew_image

    parser = argparse.ArgumentParser(
        prog="distort",
        parents=[parent_parser]
    )

    parser.add_argument(
        "image_path",
        help="file path of image to be drawn on"
    )

    parser.add_argument(
        "-o",
        "--output_path",
        help="output path of image with the modifications"
    )

    parser.add_argument(
        "-p",
        "--patch",
        nargs="+",
        action="append",
        help=(
            "area to focus on the image; x,y points\n"
            "should be a four sided polygon\n"
            "example: --patch 10,10 10,400 400,400 400,10"
        )
    )

    args = parser.parse_args(arguments)
    patch = None
    if args.patch:
        patch = [
            [int(c) for c in point.split(",")] for point in args.patch[-1]
        ]

    skew_image(
        image_path=args.image_path,
        output_path=args.output_path,
        patch=patch
    )

    return 0


def batch_skew(parent_parser, arguments):
    from .functions.skew import batch_skew

    parser = argparse.ArgumentParser(
        prog="batch_distort",
        parents=[parent_parser]
    )

    parser.add_argument(
        "batch_dir",
        help="file path of images for batch procssesing"
    )

    parser.add_argument(
        "-o",
        "--output_dir",
        required=True,
        help="output directory of distorted images"
    )

    parser.add_argument(
        "-c",
        "--compressed_path",
        help="path of the compressed file to be generated"
    )

    args = parser.parse_args(arguments)
    batch_skew(
        batch_dir=args.batch_dir,
        output_dir=args.output_dir,
        compressed_path=args.compressed_path
    )

    return 0
