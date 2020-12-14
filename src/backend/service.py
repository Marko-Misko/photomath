import cv2 as cv
import numpy as np
from flask import (
    Blueprint, request, abort, render_template
)

from backend import ALLOWED_EXTENSIONS
from src.ml.image_processor import Photomath

bp = Blueprint('service', __name__)

photomath = Photomath.create_vanilla_photomath()


@bp.route('/', methods=['GET', 'POST'])
def index():
    """
    Single page of Photomath web app. Renders simple UI on which user can
    upload a single valid image and request it's solving.

    :return: HTML page
    """
    if request.method == 'POST':
        image = request.files['image']

        if image.filename == '':
            abort(422, 'No selected file')
        if ('.' in image.filename and image.filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS) is False:
            abort(422, 'File format not supported')

        image = cv.imdecode(np.frombuffer(image.read(), np.uint8), cv.IMREAD_UNCHANGED)

        img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        expr, result = photomath.process_photo(img)
        return render_template('service.html', solution=result)

    return render_template('service.html')
