from typing import Tuple

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.datasets import mnist


class DatasetLoader:
    """
    Class for loading digits and operators dataset.
    """

    def __init__(self, data_dir: str):
        """
        Digits are loaded from MNIST dataset while operators are to
        be provided as custom appendix from `data_dir` folder containing
        all operators images.

        :param data_dir: folder containing rest of training data
        """
        self._data_dir = data_dir

    def load_data(self, batch_size: int = 64,
                  shuffle_buffer_size: int = 100) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """
        Creates training and validation dataset ratio similar that to the
        original MNIST dataset with about 6/7 od data in training set and
        1/7 data in validation set. Images are single channel grayscale,
        28x28 pixels large. Dataset is batched and shuffled.

        :param batch_size: batch size for dataset
        :param shuffle_buffer_size: parameter for shuffling dataset
        :return:
        """
        (train_X, train_y), (test_X, test_y) = mnist.load_data()

        operators_train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            self._data_dir,
            validation_split=1 / 7,
            subset="training",
            color_mode='grayscale',
            seed=123,
            image_size=(28, 28),
            batch_size=1024)
        for images, labels in tfds.as_numpy(operators_train_ds):
            train_X = np.concatenate((train_X, 255 - images.squeeze().astype('uint8')))
            train_y = np.concatenate((train_y, labels.astype('uint8') + 10))

        operators_test_ds = tf.keras.preprocessing.image_dataset_from_directory(
            self._data_dir,
            validation_split=1 / 7,
            subset="validation",
            color_mode='grayscale',
            seed=123,
            image_size=(28, 28),
            batch_size=1024)
        for images, labels in tfds.as_numpy(operators_test_ds):
            test_X = np.concatenate((test_X, 255 - images.squeeze().astype('uint8')))
            test_y = np.concatenate((test_y, labels.astype('uint8') + 10))

        train_dataset = tf.data.Dataset.from_tensor_slices((train_X, train_y))
        test_dataset = tf.data.Dataset.from_tensor_slices((test_X, test_y))
        train_dataset = train_dataset.shuffle(shuffle_buffer_size).batch(batch_size)
        test_dataset = test_dataset.batch(batch_size)

        return train_dataset, test_dataset
