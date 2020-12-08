from abc import ABC, abstractmethod
import numpy as np


class BaseDetector(ABC):

    @abstractmethod
    def detect(self, image: np.array):
        pass


class OpenCVDetector(BaseDetector):

    def __init__(self):
        pass

    def detect(self, image: np.array):
        pass
