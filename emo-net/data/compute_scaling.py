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
import tensorflow as tf
import numpy as np
import pickle
import glob
from tqdm import tqdm
from ..models.input_layers import LogMelgramLayer
from ..data.loader import AudioDataGenerator
from os.path import join
from sklearn.preprocessing import StandardScaler



def compute_scaling(dataset_base):
    train_generator = AudioDataGenerator(join(dataset_base, 'train.csv'),
                                         '/mnt/nas/data_work/shahin/EmoSet/wavs-reordered/',
                                         batch_size=1,
                                         window=None,
                                         shuffle=False,
                                         sr=16000,
                                         time_stretch=None,
                                         pitch_shift=None,
                                         save_dir=None,
                                         val_split=None,
                                         subset='train',
                                         variable_duration=True)
    train_dataset = train_generator.tf_dataset().prefetch(tf.data.experimental.AUTOTUNE)
    
    input_tensor = tf.keras.layers.Input(shape=(None,))
    input_reshaped = tf.keras.layers.Reshape(
            target_shape=(-1, ))(input_tensor)

    x = LogMelgramLayer(num_fft=512,
                                     hop_length=256,
                                     sample_rate=16000,
                                     f_min=80,
                                     f_max=8000,
                                     num_mels=64,
                                     eps=1e-6,
                                     return_decibel=True,
                                     name='trainable_stft')(input_reshaped)
    model = tf.keras.Model(inputs=input_tensor, outputs=x)
    spectrograms = []
    for batch in tqdm(train_dataset):
        spectrograms.append(np.squeeze(model.predict_on_batch(batch)))   
    spectrograms_concat = np.concatenate(spectrograms)
    mean = np.mean(spectrograms_concat)
    std = np.std(spectrograms_concat)
    mean_std = {'mean': mean, 'std': std}
    print(dataset_base, mean, std)
    with open(join(dataset_base, 'mean_std.pkl'), 'wb') as f:
        pickle.dump(mean_std, f)
        
        
if __name__=='__main__':
    datasets = glob.glob('/mnt/student/MauriceGerczuk/EmoSet/multiTaskSetup-wavs-with-test/*/')
    for dataset in datasets:
        compute_scaling(dataset)