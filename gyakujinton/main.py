import cv2
import numpy as np
from .Window import Window
from .Shape import Shape


def generate_superimposition():
    window = Window(width=400, height=400)

    square = Shape([
        [0, 0],
        [0, 100],
        [100, 100],
        [100, 0],
    ])
    window.register(square, rgb=(20, 100, 20))

    reflected_square = Shape([
        [50, 50],
        [50, 150],
        [150, 150],
        [150, 50],
    ])
    window.register(reflected_square, rgb=(255, 100, 100))

    return window.show()


def draw_on_image(image_path, points, output_path=None, color=(20, 100, 20)):
    from pathlib import Path

    if not Path(image_path).is_file():
        raise FileNotFoundError(
            "The path `{}` is not valid".format(image_path)
        )

    window = Window(image_path=image_path)
    square = Shape(points)
    window.register(square, rgb=color)

    if output_path:
        window.save(output_path)
        return

    return window.show()


def rotate_image(
    image_path,
    output_path=None,
    angle=40,
    scale=1.0,
    patch=None
):
    image = Window(image_path=image_path)
    (height, width, _) = image.window.shape

    if patch is None:
        patch = [
            [0, 0],
            [0, height],
            [height, width],
            [width, 0],
        ]

    center = (
        (patch[0][0] + patch[2][0]) / 2,
        (patch[0][1] + patch[2][1]) / 2,
    )

    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    image.window = cv2.warpAffine(
        image.window,
        matrix,
        image.window.shape[1::-1],
        flags=cv2.INTER_LINEAR
    )

    if output_path:
        image.save(output_path)
        return

    return image.show()


def skew_image(image_path, output_path=None, patch=None):
    import random

    image = Window(image_path=image_path)

    if patch is not None:
        image.window = image.window[
            patch[0][1]: patch[1][1],
            patch[0][0]: patch[3][0],
        ]

    (height, width, _) = image.window.shape

    if patch is None:
        patch = [
            (0, 0),
            (0, width),
            (width, height),
            (height, 0),
        ]

    skew_coords = []
    for point in patch:
        perc_rand = random.randint(10, 40)
        skew_coords += [(
            round(point[0] + (point[0] * (perc_rand / 100))),
            round(point[1] + (point[1] * (perc_rand / 100))),
        )]

    # convert to valid input for cv2 homography
    patch = np.array(patch)
    skew_coords = np.array(skew_coords)

    skewed_width = skew_coords[1][1]
    if skew_coords[2][0] > skew_coords[1][1]:
        skewed_width = skew_coords[2][0]

    skewed_height = skew_coords[2][1]
    if skew_coords[3][0] > skew_coords[2][1]:
        skewed_height = skew_coords[3][0]

    h, status = cv2.findHomography(patch, skew_coords)
    image.window = cv2.warpPerspective(
        image.window[patch[0][1]:patch[2][1], patch[0][0]:patch[2][0]],
        h,
        (skewed_width, skewed_height)
    )

    image.window = cv2.resize(
        image.window,
        (round(width / 1.125), round(height / 1.125))
    )

    padding = 50
    screen = Window(width=width + padding, height=height + padding)
    screen.window[
        padding:image.window.shape[0] + padding,
        padding:image.window.shape[1] + padding,
        :
    ] = image.window

    # set alpha channel
    b_channel, g_channel, r_channel = cv2.split(screen.window)
    alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 255

    # set alpha value to transparent of background is black
    for d, dimension in enumerate(screen.window):
        for p, pixel in enumerate(dimension):
            if list(pixel) == [0, 0, 0]:
                alpha_channel[d][p] = 0

    screen.window = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))

    if output_path:
        screen.save(output_path)
        return

    screen.show()
