class Shape():
    def __new__(cls, points):
        import numpy as np

        return np.array([points], np.float32)
