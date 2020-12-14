import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

from src.ml.data import DatasetLoader


class TensorflowClassifier:
    """
    Classifier for classifying digits and operators.
    Model is trained using ADAM optimizer and sparse categorical
    cross-entropy loss.
    """

    def __init__(self):
        num_classes = 16

        self.model = Sequential([
            layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(28, 28, 1)),
            layers.Conv2D(16, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Dropout(0.2),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(num_classes)
        ])

        self.model.compile(optimizer='adam',
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           metrics=['accuracy'])

        self.epochs = 15
        self.history = None
        self.classes = "0123456789()+-/*"

    def train(self, train_ds: tf.data.Dataset, test_ds: tf.data.Dataset, save_dir: str = None):
        """
        Performs training through all epochs.
        If `save_dir` is provided, represents the folder where
        the model will be saved, otherwise model is not saved.

        :param train_ds: train dataset
        :param test_ds: test dataset
        :param save_dir: folder where model will be saved
        """
        self.history = self.model.fit(
            train_ds,
            validation_data=test_ds,
            epochs=self.epochs
        )
        if save_dir:
            self.save_weights(save_dir)

    def save_weights(self, save_dir: str):
        """
        Saves model in `save_dir` folder.

        :param save_dir: folder where model will be saved
        """
        self.model.save(save_dir)

    def load_trained_weights(self, model_dir: str):
        """
        Loads pretrained model for classifying instead of training new model
        or to continue training on previous weights.

        :param model_dir: folder where the models was saved
        """
        self.model = tf.keras.models.load_model(model_dir)

    def classify(self, image: np.ndarray) -> str:
        """
        Perform classification on grayscale image.

        :param image: numpy array representing image
        :return: name of the correct class
        """
        img_array = np.expand_dims(image, axis=(0, 3)).astype(np.uint8)
        img_array = tf.image.resize(img_array, [28, 28])

        predictions = self.model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        return self.classes[np.argmax(score)]

    def get_metrics(self, save_dir: str):
        """
        Write last training and validation accuracy and loss to file.

        :param save_dir: name of file to store metrics
        """
        acc = self.history.history['accuracy']
        val_acc = self.history.history['val_accuracy']

        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']

        epochs_range = range(self.epochs)

        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.savefig(save_dir, bbox_inches='tight')


# Train classifier
if __name__ == '__main__':
    classifier_name = sys.argv[1]
    data_dir = os.path.abspath(os.path.join(__file__, '../../../data/processed'))
    dl = DatasetLoader(data_dir)
    train_ds, test_ds = dl.load_data()
    model = TensorflowClassifier()
    model_save_dir = os.path.abspath(os.path.join(__file__, f'../../../models/{classifier_name}'))
    model.train(train_ds, test_ds, model_save_dir)
    metrics_save_dir = os.path.abspath(os.path.join(__file__, '../../../metrics/{classifier_name}'))
    model.get_metrics(metrics_save_dir)
