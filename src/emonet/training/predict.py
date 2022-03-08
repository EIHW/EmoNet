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
tf.random.set_seed(42)

import time
import pandas as pd
from os.path import join
from sklearn.utils import class_weight
from .losses import *
from .metrics import *
from ..models.build_model import create_multi_task_resnets, create_multi_task_rnn, create_multi_task_networks
from ..data.loader import *
from os import makedirs
import logging
logger = logging.getLogger(__name__)


def predict(
    initial_weights='weights.h5',
    feature_extractor='cnn',
    batch_size=64,
    window=5,
    num_mels=128,
    tasks=["IEMOCAP"],
    directory='EmoSet/IEMOCAP',
    share_feature_layer=True,
    input_bn=False,
    mode='adapters',
    output='pred.csv',
    **kwargs):
    if feature_extractor in ['cnn', 'vgg16']:
        variable_duration = False if kwargs['classifier'] == 'avgpool' else True
    else:
        variable_duration = True
    #variable_duration = False


    test_generator = AudioDataGenerator(None,
                                       directory,
                                       batch_size=batch_size,
                                       window=window,
                                       shuffle=False,
                                       sr=16000,
                                       subset="test",
                                       time_stretch=None,
                                       pitch_shift=None,
                                       save_dir=None,
                                       variable_duration=variable_duration)






    x, _ = test_generator[0]
    print(x.shape)
    if not variable_duration:
        init = x.shape[1:]
    else:
        init = (None, )
    models, shared_model = create_multi_task_networks(
        init,
        feature_extractor=feature_extractor,
        initial_weights=initial_weights,
        num_mels=num_mels,
        new_tasks=[],
        new_nb_classes=[],
        mode=mode,
        input_bn=input_bn,
        learnall=False,
        share_feature_layer=share_feature_layer,
        **kwargs)

    #print(model.non_trainable_weights)
    #model.load_weights(initial_weights, by_name=True)

    filenames = list(test_generator.files)

    logger.info("Model loaded.")
    df = pd.DataFrame(columns=["filename"])
    df['filename'] = filenames
    tasks = tasks if tasks is not None else models.keys()
    print(f"Generating predictions for tasks: {tasks}")
    for task in tasks:
        model = models[task]
        model.summary()
        model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.SGD(lr=0.1), metrics=['accuracy'])
        predictions = model.predict(test_generator.tf_dataset())
        probas = predictions
        df[[f'{task}_probability_{i}' for i in range(probas.shape[1])]] = probas
    df.to_csv(output, index=False)

if __name__=="__main__":
    weights = ""
    input_dir = ""
    predict(initial_weights=weights, tasks=None, directory=input_dir, classifier="FCNAttention", N=2, factor=1, num_mels=64, window=5)

