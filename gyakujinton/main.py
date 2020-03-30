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
