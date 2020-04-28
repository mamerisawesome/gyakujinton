import numpy as np
import cv2
from gyakujinton.Window import Window


def skew_image(image_path, output_path=None, patch=None):
    import random

    image = Window(image_path=image_path)

    if patch is not None:
        image.window = image.window[
            patch[0][1]: patch[1][1],
            patch[0][0]: patch[3][0],
        ]

    (height, width, _) = image.window.shape
    original_image = image.window[:]

    if patch is None:
        patch = [
            (0, 0),
            (height, 0),
            (width, height),
            (0, width),
        ]

    all_x = [point[0] for point in patch]
    all_y = [point[1] for point in patch]

    skew_coords = []
    for point in patch:
        perc_rand = random.uniform(0.1, 0.4)
        new_x = 0
        new_y = 0

        if point[0] == min(all_x) and point[1] == min(all_y):
            new_x = round(point[0] + ((width / 2) * perc_rand))
            new_y = round(point[1] + ((height / 2) * perc_rand))

        elif point[0] > min(all_x) and point[1] > min(all_y):
            new_x = round(point[0] - ((width / 2) * perc_rand))
            new_y = round(point[1] - ((height / 2) * perc_rand))

        elif point[0] > min(all_x) and point[1] < max(all_y):
            new_x = round(point[0] - ((width / 2) * perc_rand))
            new_y = round(point[1] + ((height / 2) * perc_rand))

        elif point[0] < max(all_x) and point[1] > min(all_y):
            new_x = round(point[0] + ((width / 2) * perc_rand))
            new_y = round(point[1] - ((height / 2) * perc_rand))

        skew_coords += [(new_x, new_y)]

    # convert to valid input for cv2 homography
    patch = np.array(patch)
    skew_coords = np.array(skew_coords)

    h, status = cv2.findHomography(patch, skew_coords)
    image.window = cv2.warpPerspective(
        src=image.window,
        M=h,
        dsize=(width, height)
    )

    padding = 0
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
    output = {
        "original": {
            "corners": patch.tolist(),
            "image": original_image,
        },
        "warped": {
            "corners": skew_coords.tolist(),
            "image": image.window,
        },
    }

    if output_path:
        screen.save(output_path)
        return output

    screen.show()

    return output
