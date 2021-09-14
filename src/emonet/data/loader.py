#                    EmoNet
# ==============================================================================
# Copyright (C) 2021 Maurice Gerczuk, Shahin Amiriparian,
# Sandra Ottl, Björn Schuller: University of Augsburg. All Rights Reserved.
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
# ==============================================================================
import numpy as np
import time
import pandas as pd
import numpy as np
import itertools
import csv
import librosa
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import applications
from tensorflow.keras.utils import Sequence
from tensorflow.keras.utils import to_categorical
from sklearn.utils import class_weight
from sklearn.preprocessing import LabelEncoder
#from vgg16bn import Vgg16BN
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import StratifiedShuffleSplit
from os.path import join, dirname, basename, relpath
from os import makedirs
from math import ceil
from abc import ABC, abstractmethod
from glob import glob
from PIL import Image
import tensorflow as tf
import logging
logger = logging.getLogger(__name__)

class AudioDataGenerator(Sequence):
    def __init__(self,
                 csv_file,
                 directory,
                 batch_size=32,
                 window=1,
                 shuffle=True,
                 sr=16000,
                 time_stretch=None,
                 pitch_shift=None,
                 save_dir=None,
                 val_split=0.2,
                 val_indices=None,
                 subset='train',
                 variable_duration=False):
        self.random_state = 42
        self.variable_duration = variable_duration
        self.files = []
        self.classes = []
        with open(csv_file) as f:
            reader = csv.reader(f, delimiter=',')
            header = next(reader)
            if 'label' in header:
                label_index = header.index('label')
                logger.info(f'Setup csv "{csv_file}" contains "label" column at index {label_index}.')

            else:
                label_index = len(header) - 1
                logger.warn(f'Setup csv "{csv_file}" does not contain "label" column. Choosing last column: "{header[label_index]}" instead.')
            if 'file' in header:
                path_index = header.index('file')
                logger.info(f'Setup csv "{csv_file}" contains "file" column at index {path_index}.')

            else:
                path_index = 0
                logger.warn(f'Setup csv "{csv_file}" does not contain "file" column. Choosing first column: "{header[path_index]}" instead.')
            for line in reader:
                self.files.append(
                    join(directory, line[path_index]))
                self.classes.append(line[label_index])

        logger.info(f'Parsed {len(self.files)} audio files')
        self.val_split = val_split
        self.train_indices = None
        self.val_indices = val_indices
        self.subset = subset



        self.label_binarizer = LabelEncoder()
        self.label_binarizer.fit(self.classes)

        if self.val_split is not None and subset == 'train':
            self.__create_split()
        elif not (self.val_indices is None):
            self.__apply_split()

        self.directory = directory
        self.window = window
        self.classes = self.label_binarizer.transform(self.classes)
        if len(self.label_binarizer.classes_) > 2:
            self.categorical_classes = to_categorical(self.classes)
        else:
            self.categorical_classes = self.classes
        self.class_indices = {c: i for i, c in enumerate(self.label_binarizer.classes_) }
        logger.info(f'Class indices: {self.class_indices}')
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.time_stretch = time_stretch
        self.pitch_shift = pitch_shift
        self.save_dir = save_dir
        self.sr = sr
        np.random.seed(self.random_state)
        self.on_epoch_end()


    @staticmethod
    def load_audio(filename, label):
        raw = tf.io.read_file(filename)
        audio, sr = tf.audio.decode_wav(raw, desired_channels=1)
        audio = tf.reshape(audio, (-1,))
        return audio, label


    @staticmethod
    def random_slice(audio, label, size):
        size = tf.math.minimum(tf.shape(audio), size)
        audio = tf.image.random_crop(audio, size, seed=42)
        return audio, label

    @staticmethod
    def center_slice(audio, label, size):
        duration = tf.shape(audio)[0]
        start = duration // 2 if duration // 2 > size else 0
        audio = audio[start:start+size]
        return audio, label


    def tf_dataset(self):
        dataset = tf.data.Dataset.from_tensor_slices((self.files, self.categorical_classes))
        binary = len(self.categorical_classes.shape) < 2
        if self.shuffle:
            dataset = dataset.shuffle(len(self.files), seed=42)
        dataset = dataset.map(AudioDataGenerator.load_audio, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        #dataset = dataset.filter(lambda x, _: tf.math.count_nonzero(x) > 0)
        if self.window is not None:
            window_size = int(self.window*self.sr)
            padded_size = window_size if not self.variable_duration else None
            if self.shuffle:
                dataset = dataset.map(lambda audio, label: AudioDataGenerator.random_slice(audio, label, size=window_size), num_parallel_calls=tf.data.experimental.AUTOTUNE)
            else:
                dataset = dataset.map(lambda audio, label: AudioDataGenerator.center_slice(audio, label, size=window_size), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        else :
            padded_size = None
        padded_label_size = () if binary else (self.categorical_classes.shape[1],)
        dataset = dataset.padded_batch(self.batch_size, ((padded_size,), padded_label_size))
        #dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE).cache()
        return dataset






    def __create_split(self):
        sss = StratifiedShuffleSplit(n_splits=1, test_size=self.val_split, random_state=self.random_state)
        for train_index, test_index in sss.split(self.files, self.classes):
            self.val_indices = test_index
            self.train_indices = train_index


    def __apply_split(self):
        indices = self.train_indices if self.subset == 'train' else self.val_indices
        for index in sorted(indices, reverse=True):
                del self.files[index]
                del self.classes[index]


    def __len__(self):
        return ceil(len(self.files) / self.batch_size)
        """ if len(self.files) % self.batch_size == 0:
            return int(len(self.files) / self.batch_size)
        else:
            return int(len(self.files) / self.batch_size) + 1 """

    def __getitem__(self, index):
        # Generate indexes of the batch
        index = index % len(self)
        indices = self.indices[index * self.batch_size:min(len(self.indices), (index + 1) *
                               self.batch_size)]

        files_batch = [self.files[k] for k in indices]
        y = np.asarray([self.categorical_classes[k] for k in indices])

        # Generate data
        x = self.__data_generation(files_batch)

        return x, y

    def _set_index_array(self):
        self.indices = np.arange(len(self.files))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self._set_index_array()

    def __data_generation(self, files):
        audio_data = []

        for file in files:
            duration = librosa.core.get_duration(filename=file)

            if self.window is not None:
                stretched_window = self.window * (
                    1 + self.time_stretch
                ) if self.time_stretch is not None else self.window
                if self.shuffle:
                    start = np.random.randint(0, max(1, int(duration - stretched_window)))

                else:
                    start = duration / 2 if duration / 2 > stretched_window else 0  # take the middle chunk
                y, sr = librosa.core.load(file,
                                          offset=start,
                                          duration=min(stretched_window, duration),
                                          sr=self.sr)
                y = self.__get_random_transform(y, sr)
                end_sample = min(int(self.window * sr), int(duration * sr))
                y = y[:end_sample]
            else:
                y, sr = librosa.core.load(file, sr=self.sr)
                y = self.__get_random_transform(y, sr)

            if self.save_dir:
                rel_path = relpath(file, self.directory)
                save_path = join(self.save_dir, rel_path.wav)
                makedirs(dirname(save_path), exist_ok=True)
                librosa.output.write_wav(
                    join(self.save_dir, rel_path),
                    audio_data, sr)
            audio_data.append(y)
        if (self.window is not None) and (not self.variable_duration):
            audio_data = pad_sequences(
                audio_data, maxlen=int(self.window*self.sr), dtype='float32')
        else:
            audio_data = pad_sequences(
                audio_data, dtype='float32')

        return audio_data

    def __get_random_transform(self, y, sr):
        if self.time_stretch is not None:
            factor = np.random.normal(1, self.time_stretch)
            y = librosa.effects.time_stretch(y, factor)
        if self.pitch_shift is not None:
            steps = np.random.randint(0 - self.pitch_shift,
                                      1 + self.pitch_shift)
            y = librosa.effects.pitch_shift(y, sr, steps)
        return y


def benchmark(dataset, num_epochs=2):
    start_time = time.perf_counter()
    for epoch_num in range(num_epochs):
        for sample in dataset:
            # Performing a training step
            time.sleep(0.01)
    tf.print("Execution time:", time.perf_counter() - start_time)

def benchmark_generator(generator, num_epochs=2):
    start_time = time.perf_counter()
    for _ in range(num_epochs):
        for i in range(len(generator)):
            _ = generator[i]
            # Performing a training step
            # time.sleep(0.01)
    tf.print("Execution time:", time.perf_counter() - start_time)
