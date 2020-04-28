import cv2
from gyakujinton.Window import Window


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
