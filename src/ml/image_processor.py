import os
import sys
from typing import Tuple, Union

import cv2 as cv
import numpy as np

from src.ml.classifier import TensorflowClassifier
from src.ml.detector import OpenCVDetector
from src.ml.solver import StackSolver, StackSolverException


class Photomath:
    """
    Program for reading input image which has a simple mathematical expression
    written on it and than solving that expression.
    """

    def __init__(self, detector, classifier, solver):
        """
        Creates program instance constructed with detector, classifier and solver
        in order to perform image processing.

        :param detector: object detector
        :param classifier: multi-class classifier
        :param solver: math expression solver
        """
        self.detector = detector
        self.classifier = classifier
        self.solver = solver

    def process_photo(self, image: np.ndarray) -> Tuple[str, Union[str, int]]:
        """
        Performs expression evaluation on image.

        :param image: input image
        :return: tuple containing read expression and it's result
        """
        characters = self.detector.detect(image)

        expression = ''
        for char in characters:
            crop = cv.copyMakeBorder(char[2],
                                     10, 10, 20, 20,
                                     cv.BORDER_CONSTANT, None, 255)
            ret, thresh = cv.threshold(255 - crop, 127, 255, 0)

            kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
            dilate = cv.dilate(thresh, kernel, iterations=1)

            # cv.imshow('u', dilate)
            # cv.waitKey(0)
            expression += self.classifier.classify(dilate)

        try:
            result = self.solver.solve(expression)
        except StackSolverException as err:
            return expression, str(err)

        return expression, result

    @staticmethod
    def create_vanilla_photomath(classifier_name: str ='classifier'):
        """
        Factory method for creating simple photomath from basic components.

        :param classifier_name: folder name where the model is saved
        :return: photomath instance
        """
        detector = OpenCVDetector()
        models_dir = os.path.abspath(os.path.join(__file__, f'../../../models/{classifier_name}'))
        if os.listdir(models_dir):
            classifier = TensorflowClassifier()
            classifier.load_trained_weights(
                os.path.abspath(os.path.join(__file__, f'../../../models/{classifier_name}')))
        else:
            sys.exit(1)
        solver = StackSolver()
        photomath = Photomath(detector, classifier, solver)
        return photomath


# Image processor test
if __name__ == '__main__':
    classifier_name = sys.argv[1]
    path_to_image = sys.argv[2]
    photomath = Photomath.create_vanilla_photomath(classifier_name)
    img = cv.imread(path_to_image)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    expr, result = photomath.process_photo(img)
    print(f'Expression read is {expr} and the result is: {result}')
