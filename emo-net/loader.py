import numpy as np
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



class SequentialMultiTaskGenerator(Sequence, ABC):
    def __init__(self,
                 batch_size,
                 task_files,
                 directory=None,
                 shuffle=False,
                 **kwargs):

        self.shuffle = shuffle
        self.batch_size = batch_size
        self.class_indices = dict()
        self.classes = []
        self._task_csvs = sorted(task_files)
        self.directory = directory
        self.task_names = list(
            map(lambda x: basename(dirname(x)), self._task_csvs))
        self.frames = [pd.read_csv(csv) for csv in sorted(self._task_csvs)]
        self.generators = None
        self._setup_generators()
        masks = np.full(
            (len(self.task_names),
                sum(len(g.class_indices) for g in self.generators)), -1)
        i = 0
        for j, (tn, g) in enumerate(zip(self.task_names, self.generators)):
            for k, v in sorted(g.class_indices.items()):
                self.class_indices[f'{tn}/{k}'] = v + i
            masks[j, i:(i + len(g.class_indices))] = 0
            gen_classes = np.array(g.classes) + i
            self.classes.append(gen_classes)
            i += len(g.class_indices)
        self.classes = np.concatenate(self.classes)

        self.lengths = [len(gen) for gen in self.generators]
        self._index_array = np.arange(len(self))
        self._batchid_to_gen = np.repeat(range(len(self.generators)),
                                         self.lengths)
        self._individual_batchids = np.array(list(
            itertools.chain(*[range(l) for l in self.lengths])),
                                             dtype=int)

        self.task_masks = {
            tn: masks[i]
            for i, tn in enumerate(self.task_names)
        }
        self._masks = np.concatenate(
            [np.tile(masks[i], (l, 1)) for i, l in enumerate(self.lengths)])
        self.on_epoch_end()

    @abstractmethod
    def _setup_generators(self):
        pass

    def _set_index_array(self):
        if self.shuffle:
            np.random.shuffle(self._index_array)

    def reset_index(self):
        """Deep shuffle.
        """
        self._set_index_array()
        for g in self.generators:
            g._set_index_array()

    def on_epoch_end(self):
        self.reset_index()

    def __len__(self):
        # round up
        return sum(self.lengths)

    def get_steps_per_epoch(self):
        """Get number of steps per epoch based on batch size and
        number of images.

        Returns:
            int -- steps per epoch.
        """
        return self.__len__()

    def __getitem__(self, index):
        'Generate one batch of data'

        X, y = self.__data_generation(index)
        return X, y

    def __data_generation(self, batch_id):
        shuffled_index = self._index_array[batch_id]
        generator = self.generators[self._batchid_to_gen[shuffled_index]]
        generator_batchid = self._individual_batchids[shuffled_index]
        X, y = generator[generator_batchid]
        mask = self._masks[shuffled_index]
        y_masked = np.zeros((y.shape[0], mask.shape[0]))
        y_masked += mask
        y_masked[:,
                 min(np.where(
                     y_masked == 0)[1]):max(np.where(y_masked == 0)[1]) +
                 1] = y
        assert np.any(X) and np.any(
            y_masked), f'Bad image or label data:\n {X}\n {y}'
        assert not np.isnan(X).any(
        ), f'NaNs in input! For batch id {batch_id} and task {self.task_names[self._batchid_to_gen[shuffled_index]]}'
        assert not np.isnan(y_masked).any(
        ), f'NaNs in labels: {y_masked}\n For batch id {batch_id} and task {self.task_names[self._batchid_to_gen[shuffled_index]]}'
        return X, y_masked

class SequentialMultiTaskImageDataGenerator(SequentialMultiTaskGenerator):
    def __init__(self,
                 batch_size,
                 task_files,
                 directory=None,
                 shuffle=False,
                 generator=None,
                 img_height=224,
                 img_width=224):
        self.generator = generator
        self.img_height = img_height
        self.img_width = img_width
        super().__init__(batch_size, task_files, directory, shuffle)

    def _setup_generators(self):
        self.generators = [
            self.generator.flow_from_dataframe(dataframe=frame,
                                            directory=self.directory,
                                            x_col='file',
                                            y_col='label',
                                            target_size=(self.img_width,
                                                        self.img_height),
                                            batch_size=self.batch_size,
                                            class_mode='categorical',
                                            shuffle=self.shuffle,
                                            seed=42) for frame in self.frames
        ]


class SequentialMultiTaskAudioDataGenerator(SequentialMultiTaskGenerator):
    def __init__(self,
                 batch_size,
                 task_files,
                 directory=None,
                 shuffle=False,
                 window=1,
                 sr=16000,
                 time_stretch=None,
                 pitch_shift=None,
                 save_dir=None,):
        self.window = window
        self.sr = sr
        self.time_stretch = time_stretch
        self.pitch_shift = pitch_shift
        self.save_dir = save_dir
        super().__init__(batch_size, task_files, directory, shuffle)

    def _setup_generators(self):
        self.generators = [
            AudioDataGenerator(csv,
                 self.directory,
                 batch_size=self.batch_size,
                 window=self.window,
                 shuffle=self.shuffle,
                 sr=self.sr,
                 time_stretch=self.time_stretch,
                 pitch_shift=self.pitch_shift,
                 save_dir=self.save_dir) for csv in self._task_csvs
        ]

class MixupImageDataGenerator(Sequence):
    def __init__(self,
                 generator,
                 directory,
                 csv,
                 batch_size,
                 img_height,
                 img_width,
                 alpha=0.2,
                 subset=None,
                 shuffle=True):
        """Constructor for mixup image data generator.

        Arguments:
            generator {object} -- An instance of Keras ImageDataGenerator.
            directory {str} -- Image directory.
            batch_size {int} -- Batch size.
            img_height {int} -- Image height in pixels.
            img_width {int} -- Image width in pixels.

        Keyword Arguments:
            alpha {float} -- Mixup beta distribution alpha parameter. (default: {0.2})
            subset {str} -- 'training' or 'validation' if validation_split is specified in
            `generator` (ImageDataGenerator).(default: {None})
        """
        self.shuffle = shuffle
        self.batch_index = 0
        self.batch_size = batch_size
        self.alpha = alpha
        self.frame = pd.read_csv(csv)

        # First iterator yielding tuples of (x, y)
        self.generator = generator.flow_from_dataframe(
            dataframe=self.frame,
            directory=directory,
            x_col='file',
            y_col='label',
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=self.shuffle)

        # Number of images across all classes in image directory.
        self.samples = self.generator.samples
        self.class_indices = self.generator.class_indices
        self.classes = self.generator.classes
        self.on_epoch_end()

    def reset_index(self):
        """Reset the generator indexes array.
        """

        self.generator._set_index_array()

    def on_epoch_end(self):
        self.reset_index()

    def reset(self):
        self.batch_index = 0

    def __len__(self):
        # round up
        return (self.samples + self.batch_size - 1) // self.batch_size

    def get_steps_per_epoch(self):
        """Get number of steps per epoch based on batch size and
        number of images.

        Returns:
            int -- steps per epoch.
        """
        return self.samples // self.batch_size

    def __getitem__(self, index):
        'Generate one batch of data'

        X, y = self.__data_generation(index)

        return X, y

    def __data_generation(self, batch_id):
        if self.batch_index == 0:
            self.reset_index()

        current_index = (self.batch_index * self.batch_size) % self.samples
        if self.samples > current_index + self.batch_size:
            self.batch_index += 1
        else:
            self.batch_index = 0

        X1, y1 = self.generator[batch_id]
        permuted_batch_ids = np.random.permutation(np.arange(len(X1)))
        X2, y2 = X1[permuted_batch_ids], y1[permuted_batch_ids]

        l = np.random.beta(self.alpha, self.alpha, len(X1))
        X_l = l.reshape(len(X1), 1, 1, 1)
        y_l = l.reshape(len(X1), 1)

        X = X1 * X_l + X2 * (1 - X_l)
        y = y1 * y_l + y2 * (1 - y_l)

        return X, y


class MultiTaskMixupGenerator(Sequence):
    def __init__(self,
                 generator,
                 alpha=0.2,
                 shuffle=True):
        self.shuffle = shuffle
        self.alpha = alpha

        # First iterator yielding tuples of (x, y)
        self.generator = generator
        self.class_indices = self.generator.class_indices
        self.classes = self.generator.classes
        self.on_epoch_end()

    def reset_index(self):
        """Reset the generator indexes array.
        """
        self.generator._set_index_array()

    def on_epoch_end(self):
        self.reset_index()


    def __len__(self):
        return len(self.generator)

    def __getitem__(self, index):
        'Generate one batch of data'

        X, y = self.__data_generation(index)

        return X, y

    def __data_generation(self, batch_id):

        X1, y1 = self.generator[batch_id]
        while True:
            random_batch_id = np.random.randint(0, len(self))
            if batch_id == random_batch_id:
                continue
            X2, y2 = self.generator[random_batch_id]
            if y2.shape[0] < y1.shape[0]:
                continue
            X2, y2 = X2[:X1.shape[0], :], y2[:y1.shape[0], :] 
            mask_indices1 = np.where(y1 != -1)
            mask_indices2 = np.where(y2 != -1)
            if np.array_equal(mask_indices1, mask_indices2):
                continue
            else: 
                break

        y1[mask_indices2] = 0
        l = np.random.beta(self.alpha, self.alpha, len(X1))
        X_l = l.reshape(len(X1), *[1]*(len(X1.shape)-1))
        y_l = l.reshape(len(X1), 1)
        X = X1 * X_l + X2 * (1 - X_l)
        _y = y1 * y_l + y2 * (1 - y_l)
        _y = np.clip(_y, -1, 1)
        y = np.full_like(_y, -1)
        y[mask_indices1] = _y[mask_indices1]
        y[mask_indices2] = _y[mask_indices2]
        return X, y