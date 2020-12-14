# photomath_junior

Photomath_junior is simple web app enabling user to
pass a photo of a math expression and get the result.

Service can be run as Flask application from package
src/backend. The user is presented with simple UI which asks him
to send an image of math expression to server. The server backend
is accessed through an image processor interface exposing method `process_image(image)`.
`process_image(image)` takes a grayscale image of a mathematical expression and
passes it through internal pipeline consisting of object detection which
detects digits and operators on an image then object classification which
gives back which mathematical symbols correspond to the detected objects and finally
and finally a math solver which solves the expression determined by these
symbols.

If you wish to run the program as Flask application, first set
**FLASK_APP** and **FLASK_ENV** environment variables and then
run the server with `flask run`.

If you wish to start the image processor locally run the following script,
(in this example working directory is photomath root directory)
> python -m src.ml.image_processor `{classifier_name}` `{path_to_image}`

where `{path_to_image}` should be an absolute path to the image you want to process.
`{classifier_name}` is a path relative to `models/` folder which should be inside root directory
of the project: this means that both of these ways to run a Photomath_junior depend on
photomath directory containing directory `models/` with saved models you want to use for
classification in its subdirectories.

This means that the models should first be trained.
This is done by invoking classifier script (in this example working directory is photomath root directory)
> python -m src.ml.classifier `classifier_name`

Invoking this program starts the training, using data from `data/processed` directory + MNIST dataset
from Tensorflow to train a digit & operator classifier, which will then save this model weights to 
`models/{classifier_name}` and the training metrics to `metrics/{classifier_name}.png`

`data/processed` directory should therefore contain images of operators and parentheses, because the
rest of the data (digits) is borrowed from MNIST.

As far as the problems concerned, the biggest one is for sure similarities between 1 and / which causes
most confusion to model. 
Improvements should be to increase dataset size, use data augmentation and maybe even to consider introducing 
another model, or some heuristics in order to determine if the model should consider classifying 
object as either 1 or /.
