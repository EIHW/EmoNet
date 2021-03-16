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


def evaluate(
    initial_weights='weights.h5',
    feature_extractor='cnn',
    batch_size=64,
    window=5,
    num_mels=128,
    task="",
    directory='EmoSet/IEMOCAP',
    val_csv='val.csv',
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
   
    
    val_generator = AudioDataGenerator(val_csv,
                                       directory,
                                       batch_size=batch_size,
                                       window=window,
                                       shuffle=False,
                                       sr=16000,
                                       time_stretch=None,
                                       pitch_shift=None,
                                       save_dir=None,
                                       variable_duration=variable_duration)
  
    val_dataset = val_generator.tf_dataset()

    


    
    x, _ = val_generator[0]
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
    model = models[task]
    #model.load_weights(initial_weights, by_name=True)
   
    model.summary()
    #print(model.non_trainable_weights)
    #model.load_weights(initial_weights, by_name=True)

    
    metric_callback = ClassificationMetricCallback(
        validation_generator=val_dataset.prefetch(tf.data.experimental.AUTOTUNE), dataset_name='Test', labels=val_generator.class_indices, true=val_generator.categorical_classes)
   
    
    
    filenames = list(map(lambda x: join(*(x.split('/')[-4:])), val_generator.files))
    index_to_class = {v: k for k, v in val_generator.class_indices.items()}

    logger.info("Model loaded.")
    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.SGD(lr=0.1), metrics=['accuracy'])
    metric_callback.set_model(model)
    x = model.evaluate(val_generator.tf_dataset(),
                        # use_multiprocessing=True,
                        # max_queue_size=n_workers * 2,
                        verbose=1,
                        callbacks=[
                            metric_callback
                        ])
    metric_callback.on_epoch_end(epoch=1)
    predictions = model.predict(val_generator.tf_dataset())
    probas = predictions
    if predictions.shape[1] > 1:
        predictions = list(map(lambda x: index_to_class[x], np.argmax(predictions, axis=-1)))
    else:
        predictions = list(map(lambda x: index_to_class[x], np.squeeze(np.where(predictions < 0.5, 0, 1))))
    true = list(map(lambda x: index_to_class[x], val_generator.classes))
    columns = ['filename', *[f'probability_{index_to_class[i]}' for i in range(probas.shape[1])], 'pred_label', 'true_label']
    df = pd.DataFrame(columns=columns)
    df['filename'] = filenames
    df['pred_label'] = predictions
    df['true_label'] = true
    df[[f'probability_{index_to_class[i]}' for i in range(probas.shape[1])]] = probas
    df.to_csv(output, index=False)
    
