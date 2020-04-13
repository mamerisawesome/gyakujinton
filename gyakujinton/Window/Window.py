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
            dims = (height, width, 3)
            self.window = np.zeros(dims, dtype="uint8")
            self.window.fill(0)
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

    def show(self, window_size=None):
        try:
            cv2.imshow(self.name, self.canvas)
        except AttributeError:
            cv2.imshow(self.name, self.window)

        if window_size:
            cv2.namedWindow(self.name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.name, *window_size)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def save(self, output_path):
        try:
            cv2.imwrite(output_path, self.canvas)
        except AttributeError:
            cv2.imwrite(output_path, self.window)
