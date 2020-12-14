from typing import Tuple, List

import cv2 as cv
import numpy as np


class OpenCVDetector:
    """
    Object detector, implemented as wrapper around OpenCVs contour
    detector.
    """

    @staticmethod
    def detect(image: np.array, sort: bool = True,
               min_area: int = None) -> List[Tuple[Tuple[int, int],
                                                   Tuple[int, int],
                                                   np.ndarray]]:
        """
        Detect the objects on `image` and return list of tuples.
        Each tuple contains 3 elements: upper left corner, bottom right corner
        and cropped part of the image bounded by those two corners.
        Objects can sorted in order as they show up in the picture from
        top left to bottom right.

        :param image: numpy array representing input image
        :param sort: if True sort objects
        :param min_area: minimum contour area to be recognized as valid object
        :return: list of coordinates and bounding boxes around detected objects
        """

        ret, thresh = cv.threshold(image, 127, 255, 0)

        contours, hierarchy = cv.findContours(thresh, 3, 2)

        characters = []
        for i, h in enumerate(hierarchy[0]):
            if h[3] == 0:
                cnt = contours[i]
                x, y, w, h = cv.boundingRect(cnt)
                characters.append(((x, y), (x + w, y + h), image[y:y + h, x:x + w]))

        if min_area:
            characters = list(filter(lambda x: (x[1][0] - x[0][0]) * (x[1][1] - x[0][1] > min_area), characters))

        if sort:
            max_height = max(map(lambda x: x[1][1] - x[0][1], characters))
            nearest = max_height * 1.4
            characters.sort(key=lambda r: [int(nearest * round(float(r[0][1]) / nearest)),
                                           r[0][0]])

        return characters
