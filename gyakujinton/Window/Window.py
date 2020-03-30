import cv2
import numpy as np


class Window():
    def __init__(
        self,
        name="Gyaku Jinton",
        width=512,
        height=512,
        image_path=None
    ):
        self.name = name

        if not image_path:
            dims = (width, height, 3)
            self.window = np.zeros(dims, dtype="uint8")
        else:
            self.window = cv2.imread(image_path)

    def register(self, points, rgb=(0, 0, 0), thickness=3):
        self.canvas = cv2.polylines(
            img=self.window,
            pts=points,
            isClosed=True,
            color=rgb,  # in rgb
            thickness=3
        )

    def show(self):
        cv2.imshow(self.name, self.canvas)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def save(self, output_path):
        cv2.imwrite(output_path, self.canvas)
