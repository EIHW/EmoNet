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

import tensorflow.keras.backend as K
import h5py
from .input_layers import *
from .adapter_rnn import *
from .adapter_resnet import *
from .attention import *
from ..utils import array_list_equal

import logging
logger = logging.getLogger(__name__)



def avgpool(x):
    x = tf.keras.layers.AveragePooling2D((8, 8))(x)
    x = tf.keras.layers.Flatten()(x)
    return x

def global_avgpool(x):
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Flatten()(x)
    return x


def infer_tasks_from_weightfile(initial_weights):
    with h5py.File(initial_weights) as f:
        base_tasks = []
        base_nb_classes = []
        for k in f['model_weights']:
            prefices = ('activation', 'add', 'average_pooling',
                        'batch_normalization', 'bidirectional', 'concat', 'conv2d', 'dropout', 'attention',
                        'dense', 'flatten', 'input', 'lambda',
                        'normalization2d', 'reshape', 'trainable_stft', 'apply_zero_mask', 'core', 'lstm', 'masking', 'permute', 'pooled', 'seq', 'zero_mask', 'adapter', 'mask', 'expand', 'mfcc', 'downpool')
            skip_layer = any([prefix in k for prefix in prefices])
            if not skip_layer:
                task = k
                classes = _find_n_classes(f['model_weights'][k], k)
                #classes = f['model_weights'][k]['softmax'][k]['softmax']['kernel:0'].shape[1]
                logger.info(f'Found task {k} with {classes} classes.')
                base_tasks.append(task)
                base_nb_classes.append(classes)
    return base_tasks, base_nb_classes


def _find_n_classes(weight_dict, task):
    if 'sigmoid' in weight_dict:
        return 2
    elif 'kernel:0' in weight_dict:
        output_shape = weight_dict['kernel:0'].shape[1]
        if output_shape == 1: # binary
            output_shape += 1
        return output_shape
    elif 'softmax' in weight_dict:
        return _find_n_classes(weight_dict['softmax'], task)
    elif task in weight_dict:
        return _find_n_classes(weight_dict[task], task)


def input_features_and_mask(audio_in, num_fft=1024, hop_length=512, sample_rate=16000, f_min=20, f_max=8000, num_mels=128, eps=1e-6, return_decibel=False, num_mfccs=None):
    input_features = LogMelgramLayer(num_fft=num_fft,
                                     hop_length=hop_length,
                                     sample_rate=sample_rate,
                                     f_min=f_min,
                                     f_max=f_max,
                                     num_mels=num_mels,
                                     eps=eps,
                                     return_decibel=return_decibel,
                                     name='trainable_stft')
    x = input_features(audio_in)
    mask = ComputeMask(input_features.num_fft,
                       input_features.hop_length)(audio_in)
    if num_mfccs is not None:
        x = MFCCLayer(num_mfccs=num_mfccs)(x)
    return x, mask


def create_multi_task_networks(input_dim, feature_extractor='cnn',
        initial_weights=None,
        base_nb_classes=None,
        learnall=True,
        num_mels=128,
        base_tasks=None,
        new_tasks=None,
        new_nb_classes=None,
        mode=None,
        random_noise=None,
        input_bn=False,
        share_feature_layer=True,
        base_weight_decays=None,
        new_weight_decays=None,
         **kwargs):
    if feature_extractor == 'cnn':
        return create_multi_task_resnets(input_dim=input_dim, mode=mode, num_mels=num_mels, initial_weights=initial_weights, base_nb_classes=base_nb_classes, base_weight_decays=base_weight_decays, new_weight_decays=new_weight_decays, learnall=learnall, base_tasks=base_tasks, new_tasks=new_tasks, new_nb_classes=new_nb_classes, random_noise=random_noise, input_bn=input_bn, share_feature_layer=share_feature_layer, **kwargs)
    elif feature_extractor == 'rnn':
        return create_multi_task_rnn(input_dim=input_dim, mode=mode, num_mels=num_mels, initial_weights=initial_weights, base_nb_classes=base_nb_classes, learnall=learnall, base_tasks=base_tasks, new_tasks=new_tasks, new_nb_classes=new_nb_classes, random_noise=random_noise, input_bn=input_bn, share_feature_layer=share_feature_layer, **kwargs)
    elif feature_extractor == 'vgg16':
        return create_multi_task_vgg16(input_dim=input_dim,
                        tasks=base_tasks,
                        num_mels=num_mels,
                        nb_classes=base_nb_classes,
                        random_noise=random_noise,
                        initial_weights=initial_weights,
                        share_feature_layer=share_feature_layer,
                        **kwargs)
    elif feature_extractor == 'fusion':
        return create_multi_task_fusion(input_dim=input_dim, mode=mode, num_mels=num_mels, initial_weights=initial_weights, base_nb_classes=base_nb_classes, learnall=learnall, base_tasks=base_tasks, new_tasks=new_tasks, new_nb_classes=new_nb_classes, random_noise=random_noise, input_bn=input_bn, share_feature_layer=share_feature_layer, **kwargs)
        

def create_multi_task_fusion(input_dim,filters=32,
                            factor=1,
                            N=4,
                            hidden_dim=512,
                            cell='lstm',
                            number_of_layers=2,
                            down_pool=8,
                            bidirectional=False,
                            num_mels=128,
                            learnall=True,
                            learnall_classifier=True,
                            mode='adapters',
                            dropout1=0,
                            dropout2=0,
                            rnn_dropout=0.2,
                            base_tasks=['EMO-DB', 'GEMEP'],
                            new_tasks=None,
                            base_nb_classes=[6, 10],
                            new_nb_classes=None,
                            initial_weights=None,
                            random_noise=0.1,
                            reuse_batchnorm=False,
                            input_bn=False, 
                            share_feature_layer=False):
    channel_axis = -1
    if base_tasks is None:
        assert initial_weights is not None, f'Either base tasks or initial weights have to be specified!'
        logger.info(
            'Trying to determine trained tasks from initial weights...')
        base_tasks, base_nb_classes = infer_tasks_from_weightfile(
            initial_weights)

    input_tensor = tf.keras.layers.Input(shape=input_dim)
    
    input_reshaped = tf.keras.layers.Reshape(
        target_shape=(-1, ))(input_tensor)
    

    x, mask = input_features_and_mask(input_reshaped, num_fft=1024,
                                      hop_length=512,
                                      sample_rate=16000,
                                      f_min=20,
                                      f_max=8000,
                                      num_mels=num_mels,
                                      eps=1e-6,
                                      return_decibel=True,
                                      num_mfccs=None)
    
    adapter_rnn = RNNWithAdapters(K.int_shape(x),
                                  hidden_size=hidden_dim,
                                  learnall=learnall,
                                  dropout=rnn_dropout,
                                  input_projection_factor=1,
                                  adapter_projection_factor=4,
                                  bidirectional=bidirectional,
                                  layers=number_of_layers,
                                  recurrent_cell=cell,
                                  input_bn=input_bn,
                                  downpool=down_pool,
                                  input_projection=False,
                                  # tasks=base_tasks,
                                  tasks=base_tasks if mode == 'adapters' else [],
                                  share_feature_layer=share_feature_layer)
    if down_pool is not None:
        mask = PoolMask((down_pool,))(mask)


    expand_dims = tf.keras.layers.Lambda(
        lambda x: tf.expand_dims(x, 3), name='expand_input_dims')
    x_resnet = expand_dims(x)
    x_resnet = tf.keras.layers.Permute((2, 1, 3))(x_resnet)
    x_rnn = x
    
    
    adapter_resnet = ResNet(filters=filters,
                            factor=factor,
                            N=N,
                            learnall=learnall,
                            dropout1=dropout1,
                            dropout2=dropout2,
                            # tasks=base_tasks,
                            tasks=base_tasks if mode == 'adapters' else [],
                            input_bn=input_bn)

    if new_tasks is not None:
        really_new_tasks = [t for t in new_tasks if t not in base_tasks]
        new_nb_classes = [
            c for c, t in zip(new_nb_classes, new_tasks) if t not in base_tasks
        ]
    else:
        really_new_tasks = []
        new_nb_classes = []

    task_models = {}
    outputs = []
    attention2d = None
    for task, classes in zip(base_tasks, base_nb_classes):
        logger.info(f'Building model for {task} with {classes} classes...')
        adapters_in = task if mode == 'adapters' else None
        y_resnet = adapter_resnet(x_resnet, task=adapters_in)
        y_rnn = adapter_rnn(x_rnn, task=adapters_in, mask=mask)
        
        
        if attention2d is None:
            attention2d = Attention2Dtanh(lmbda=0.3, mlp_units=K.int_shape(y_resnet)[-1], name=f'core_2d_attention', trainable=not(mode=='adapters'))
        if not share_feature_layer and mode == 'adapters':
            attention2d = Attention2Dtanh(name=f'{task}_2d_attention', mlp_units=K.int_shape(y_resnet)[-1], trainable=True)
        y_resnet = attention2d(y_resnet)

        y = tf.keras.layers.Concatenate()([y_resnet, y_rnn])
            
        y = tf.keras.layers.Dense(K.int_shape(y)[-1]//2, activation=None, name=f'{task}_dense')(y)
        y = tf.keras.layers.BatchNormalization(name=f'{task}_dense_batchnorm')(y)
        y = tf.keras.layers.Activation('relu', name=f'{task}_dense_relu')(y)    
        y = tf.keras.layers.Dropout(0.2)(y)
        if classes == 2:
            y = tf.keras.layers.Dense(1, activation='sigmoid', name=task)(y)
        else:
            y = tf.keras.layers.Dense(
                classes, activation='softmax', name=task)(y)
        model = tf.keras.Model(inputs=input_tensor, outputs=y)
        outputs.append(y)
        task_models[task] = model

    if really_new_tasks is not None and new_nb_classes is not None:
        for task, classes in zip(really_new_tasks, new_nb_classes):
            logger.info(f'Building model for {task} with {classes} classes...')
            adapter_resnet._add_new_task(task)
            adapter_rnn._add_new_task(task)
            y_resnet = adapter_resnet(x, task)
            y_rnn = adapter_rnn(x_rnn, task, mask=mask)
        
            if attention2d is None:
                attention2d = Attention2Dtanh(lmbda=0.3, mlp_units=K.int_shape(y_resnet)[-1], name=f'core_2d_attention', trainable=not(mode=='adapters'))
            if not share_feature_layer and mode == 'adapters':
                attention2d = Attention2Dtanh(lmbda=0.3, name=f'{task}_2d_attention', mlp_units=K.int_shape(y_resnet)[-1], trainable=True)
            y_resnet = attention2d(y_resnet)
            
            y = tf.keras.layers.Concatenate()([y_resnet, y_rnn])
                
            y = tf.keras.layers.Dense(K.int_shape(y)[-1]//2, activation=None, name=f'{task}_dense')(y)
            y = tf.keras.layers.BatchNormalization(name=f'{task}_dense_batchnorm')(y)
            y = tf.keras.layers.Activation('relu', name=f'{task}_dense_relu')(y)    
            y = tf.keras.layers.Dropout(0.2)(y)
            if classes == 2:
                y = tf.keras.layers.Dense(
                    1, activation='sigmoid', name=task)(y)
            else:
                y = tf.keras.layers.Dense(
                    classes, activation='softmax', name=task)(y)
            model = tf.keras.Model(inputs=input_tensor, outputs=y)
            outputs.append(y)
            task_models[task] = model
    shared_model = tf.keras.Model(inputs=input_tensor, outputs=outputs)

    if initial_weights is not None:
        shared_model, task_models = load_and_assert_loaded(shared_model, task_models, initial_weights)
        
    return task_models, shared_model

def create_multi_task_resnets(input_dim,
                              filters=32,
                              factor=1,
                              N=4,
                              num_mels=128,
                              learnall=True,
                              learnall_classifier=True,
                              mode='adapters',
                              dropout1=0,
                              dropout2=0,
                              rnn_dropout=0.2,
                              base_tasks=['EMO-DB', 'GEMEP'],
                              new_tasks=None,
                              base_weight_decays=None,
                              new_weight_decays=None,
                              base_nb_classes=[6, 10],
                              new_nb_classes=None,
                              initial_weights=None,
                              random_noise=0.1,
                              classifier='rnn',
                              input_bn=False, 
                              share_feature_layer=False):
    
    channel_axis = -1
    if base_tasks is None:
        assert initial_weights is not None, f'Either base tasks or initial weights have to be specified!'
        logger.info(
            'Trying to determine trained tasks from initial weights...')
        base_tasks, base_nb_classes = infer_tasks_from_weightfile(
            initial_weights)

    base_weight_decays = base_weight_decays if base_weight_decays is not None else [1e-6]*len(base_tasks)
    
    if new_tasks is not None:
        really_new_tasks = [t for t in new_tasks if t not in base_tasks]
        new_nb_classes = [
            c for c, t in zip(new_nb_classes, new_tasks) if t not in base_tasks
        ]
    else:
        really_new_tasks = []
        new_nb_classes = []

    new_weight_decays = new_weight_decays if new_weight_decays is not None else [1e-6]*len(really_new_tasks)
    
    # check if batchnorm should be reused
    if len(new_tasks) != 1:
        reuse_batchnorm = False
    elif new_tasks[0] in base_tasks and len(base_tasks) > 1:
        reuse_batchnorm = False
    else:
        reuse_batchnorm = True
    print(reuse_batchnorm)
    task_models = {}
    outputs = []
    adapter_rnn = None
    attention2d = None
    input_tensor = tf.keras.layers.Input(shape=input_dim)
    variable_duration = not (classifier == 'avgpool')
    if variable_duration:
        input_reshaped = tf.keras.layers.Reshape(
            target_shape=(-1, ))(input_tensor)
    else:
        input_reshaped = tf.keras.layers.Reshape(
            target_shape=(input_dim[0], ))(input_tensor)

    x, mask = input_features_and_mask(input_reshaped, num_fft=1024,
                                      hop_length=512,
                                      sample_rate=16000,
                                      f_min=20,
                                      f_max=8000,
                                      num_mels=num_mels,
                                      eps=1e-6,
                                      return_decibel=False,
                                      num_mfccs=None)

    pooled_mask = PoolMask((8,))(mask)

    expand_dims = tf.keras.layers.Lambda(
        lambda x: tf.expand_dims(x, 3), name='expand_input_dims')
    x = expand_dims(x)
    x = tf.keras.layers.Permute((2, 1, 3))(x)
    adapter_resnet = ResNet(filters=filters,
                            factor=factor,
                            N=N,
                            learnall=learnall,
                            dropout1=dropout1,
                            dropout2=dropout2,
                            weight_decays=base_weight_decays,
                            reuse_batchnorm=reuse_batchnorm,
                            tasks=base_tasks if mode == 'adapters' else [],
                            input_bn=input_bn)

    
    for task, classes in zip(base_tasks, base_nb_classes):
        logger.info(f'Building model for {task} with {classes} classes...')
        adapters_in_resnet = task if mode == 'adapters' else None
        y = adapter_resnet(x, task=adapters_in_resnet)
        
        if initial_weights is not None and classifier == 'avgpool':  # might need new last dense layer
            name = f'{task}_1'
        else:
            name = task
        if classifier == 'avgpool':
            y = avgpool(y)
        elif classifier == 'FCNAttention':
            if attention2d is None:
                attention2d = Attention2Dtanh(lmbda=0.3, mlp_units=K.int_shape(y)[-1], name=f'core_2d_attention', trainable=learnall_classifier)
            if not share_feature_layer and mode == 'adapters':
                attention2d = Attention2Dtanh(name=f'{task}_2d_attention', mlp_units=K.int_shape(y)[-1], trainable=True)
            y = attention2d(y)

        elif classifier == 'rnn':
            if adapter_rnn is None:
                adapter_rnn = RNNWithAdapters(K.int_shape(y),
                                              hidden_size=K.int_shape(y)[-1],
                                              learnall=learnall_classifier,
                                              dropout=rnn_dropout,
                                              input_projection_factor=4,
                                              adapter_projection_factor=4,
                                              share_feature_layer=share_feature_layer,
                                              # tasks=base_tasks,
                                              tasks=base_tasks if mode == 'adapters' else [])
            
            y = adapter_rnn(y, adapters_in_resnet, mask=pooled_mask)
        y = tf.keras.layers.Dense(K.int_shape(y)[-1]//2, activation=None, name=f'{name}_dense')(y)
        y = tf.keras.layers.BatchNormalization(name=f'{name}_dense_batchnorm')(y)
        y = tf.keras.layers.Activation('relu', name=f'{name}_dense_relu')(y)    
        y = tf.keras.layers.Dropout(0.2)(y)
        if classes == 2:
            y = tf.keras.layers.Dense(1, activation='sigmoid', name=name)(y)
        else:
            y = tf.keras.layers.Dense(
                classes, activation='softmax', name=name)(y)
        model = tf.keras.Model(inputs=input_tensor, outputs=y)
        outputs.append(y)
        task_models[task] = model

    if really_new_tasks is not None and new_nb_classes is not None:
        for task, classes, weight_decay in zip(really_new_tasks, new_nb_classes, new_weight_decays):
            logger.info(f'Building model for {task} with {classes} classes...')
            adapter_resnet._add_new_task(task, weight_decay=weight_decay)
            y = adapter_resnet(x, task)
        
            if classifier == 'avgpool':
                y = avgpool(y)
            elif classifier == 'rnn':
                adapter_rnn._add_new_task(task)
                y = adapter_rnn(y, task)
            elif classifier == 'FCNAttention':
                if attention2d is None:
                    attention2d = Attention2Dtanh(lmbda=0.3, mlp_units=K.int_shape(y)[-1], name=f'core_2d_attention', trainable=learnall_classifier)
                if not share_feature_layer and mode == 'adapters':
                    attention2d = Attention2Dtanh(lmbda=0.3, name=f'{task}_2d_attention', mlp_units=K.int_shape(y)[-1], trainable=True)
                y = attention2d(y)
                


                
            y = tf.keras.layers.Dense(K.int_shape(y)[-1]//2, activation=None, name=f'{task}_dense')(y)
            y = tf.keras.layers.BatchNormalization(name=f'{task}_dense_batchnorm')(y)
            y = tf.keras.layers.Activation('relu', name=f'{task}_dense_relu')(y)    
            y = tf.keras.layers.Dropout(0.2)(y)
            if classes == 2:
                y = tf.keras.layers.Dense(
                    1, activation='sigmoid', name=task)(y)
            else:
                y = tf.keras.layers.Dense(
                    classes, activation='softmax', name=task)(y)
            model = tf.keras.Model(inputs=input_tensor, outputs=y)
            outputs.append(y)
            task_models[task] = model
    shared_model = tf.keras.Model(inputs=input_tensor, outputs=outputs)

    if initial_weights is not None:
        shared_model, task_models = load_and_assert_loaded(shared_model, task_models, initial_weights)
        
    return task_models, shared_model


def create_multi_task_vgg16(input_dim,
                        tasks=['EMO-DB', 'GEMEP'],
                        nb_classes=[6, 10],
                        random_noise=0.1,
                        num_mels=128,
                        classifier='attention2d',
                        dropout=0.2,
                        initial_weights=None,
                        share_feature_layer=False,
                        freeze_up_to=None):
    channel_axis = -1
    if tasks is None:
        assert initial_weights is not None, f'Either base tasks or initial weights have to be specified!'
        logger.info(
            'Trying to determine trained tasks from initial weights...')
        base_tasks, base_nb_classes = infer_tasks_from_weightfile(
            initial_weights)

    input_tensor = tf.keras.layers.Input(shape=input_dim)
    variable_duration = not (classifier == 'avgpool')
    if variable_duration:
        input_reshaped = tf.keras.layers.Reshape(
            target_shape=(-1, ))(input_tensor)
    else:
        input_reshaped = tf.keras.layers.Reshape(
            target_shape=(input_dim[0], ))(input_tensor)

    x, mask = input_features_and_mask(input_reshaped, num_fft=1024,
                                      hop_length=512,
                                      sample_rate=16000,
                                      f_min=20,
                                      f_max=8000,
                                      num_mels=num_mels,
                                      eps=1e-6,
                                      return_decibel=True,
                                      num_mfccs=None)

    pooled_mask = PoolMask((8,))(mask)

    expand_dims = tf.keras.layers.Lambda(
        lambda x: tf.expand_dims(x, 3), name='expand_input_dims')
    x = expand_dims(x)
    x = tf.keras.layers.Permute((2, 1, 3))(x)
    x = tf.keras.layers.Convolution2D(3, 1, activation='relu', name='learn_colourmapping', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    vgg16 = tf.keras.applications.VGG16(include_top=False, weights='imagenet', pooling=None)
    #vgg16 = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', pooling=None)
    if freeze_up_to is not None:
        for layer in vgg16.layers[:freeze_up_to]:
            layer.trainable = False
    else:
        for layer in vgg16.layers:
            layer.trainable = False
    task_models = {}
    outputs = []
    adapter_rnn = None
    attention2d = None
    for task, classes in zip(tasks, nb_classes):
        logger.info(f'Building model for {task} with {classes} classes...')
        y = vgg16(x)
        
        if initial_weights is not None and classifier == 'avgpool':  # might need new last dense layer
            name = f'{task}_1'
        else:
            name = task
        if classifier == 'avgpool':
            y = global_avgpool(y)


        if classifier == 'FCNAttention':
            if attention2d is None:
                attention2d = Attention2Dtanh(lmbda=0.3, mlp_units=K.int_shape(y)[-1], name=f'core_2d_attention', trainable=True)
            if not share_feature_layer:
                attention2d = Attention2Dtanh(name=f'{task}_2d_attention', mlp_units=K.int_shape(y)[-1], trainable=True)
            y = attention2d(y)
            
        if classifier == 'rnn':
            if adapter_rnn is None:
                adapter_rnn = RNNWithAdapters(K.int_shape(y),
                                              hidden_size=K.int_shape(y)[-1],
                                              learnall=True,
                                              dropout=dropout,
                                              input_projection_factor=4,
                                              adapter_projection_factor=4,
                                              share_feature_layer=share_feature_layer,
                                              # tasks=base_tasks,
                                              tasks=[])
            
            y = adapter_rnn(y, None, mask=pooled_mask)

        y = tf.keras.layers.Dense(K.int_shape(y)[-1]//2, activation=None, name=f'{task}_dense')(y)
        y = tf.keras.layers.BatchNormalization(name=f'{task}_dense_batchnorm')(y)
        y = tf.keras.layers.Activation('relu', name=f'{task}_dense_relu')(y)    
        y = tf.keras.layers.Dropout(dropout)(y)
        if classes == 2:
            y = tf.keras.layers.Dense(1, activation='sigmoid', name=name)(y)
        else:
            y = tf.keras.layers.Dense(
                classes, activation='softmax', name=name)(y)
        model = tf.keras.Model(inputs=input_tensor, outputs=y)
        outputs.append(y)
        task_models[task] = model

    shared_model = tf.keras.Model(inputs=input_tensor, outputs=outputs)

    if initial_weights is not None:
        shared_model, task_models = load_and_assert_loaded(shared_model, task_models, initial_weights)
        
    return task_models, shared_model


def create_multi_task_rnn(input_dim,
                          num_mels=128,
                          num_mfccs=40,
                          hidden_dim=512,
                          cell='lstm',
                          number_of_layers=2,
                          down_pool=8,
                          mode='adapters',
                          bidirectional=False,
                          learnall=True,
                          dropout=0.2,
                          base_tasks=['EMO-DB', 'GEMEP'],
                          new_tasks=[],
                          base_nb_classes=[6, 10],
                          new_nb_classes=None,
                          initial_weights=None,
                          random_noise=0.1,
                          input_bn=False,
                          share_feature_layer=False,
                          use_attention=True,
                          share_attention=False,
                          input_projection=True):
    channel_axis = -1
    if base_tasks is None:
        assert initial_weights is not None, f'Either base tasks or initial weights have to be specified!'
        
        logger.info(
            'Trying to determine trained tasks from initial weights...')
        base_tasks, base_nb_classes = infer_tasks_from_weightfile(
            initial_weights)

    input_tensor = tf.keras.layers.Input(shape=input_dim)
    input_reshaped = tf.keras.layers.Reshape(target_shape=(-1, ))(input_tensor)

    x, mask = input_features_and_mask(input_reshaped,
                                      num_fft=1024,
                                      hop_length=512,
                                      num_mfccs=num_mfccs,
                                      sample_rate=16000,
                                      f_min=20,
                                      f_max=8000,
                                      num_mels=num_mels,
                                      eps=1e-6,
                                      return_decibel=num_mels is None)
    adapter_rnn = RNNWithAdapters(K.int_shape(x),
                                  hidden_size=hidden_dim,
                                  learnall=learnall,
                                  dropout=dropout,
                                  input_projection_factor=1,
                                  adapter_projection_factor=4,
                                  bidirectional=bidirectional,
                                  layers=number_of_layers,
                                  recurrent_cell=cell,
                                  input_bn=input_bn,
                                  downpool=down_pool,
                                  input_projection=input_projection,
                                  use_attention=use_attention,
                                  share_attention=share_attention,
                                  # tasks=base_tasks,
                                  tasks=base_tasks if mode == 'adapters' else [],
                                  share_feature_layer=share_feature_layer)
    if down_pool is not None:
        mask = PoolMask((down_pool,))(mask)


    if new_tasks is not None:
        really_new_tasks = [t for t in new_tasks if t not in base_tasks]
        new_nb_classes = [
            c for c, t in zip(new_nb_classes, new_tasks) if t not in base_tasks
        ]
    else:
        really_new_tasks = []
        new_nb_classes = []

    task_models = {}
    outputs = []
    for task, classes in zip(base_tasks, base_nb_classes):
        logger.info(f'Building model for {task} with {classes} classes...')
        #y = adapter_resnet(x, task)
        adapters_in = task if mode == 'adapters' else None
        y = adapter_rnn(x, adapters_in, mask=mask)
        if initial_weights is not None:  # might need new last dense layer
            name = f'{task}_1'
        else:
            name = task
        y = tf.keras.layers.Dense(K.int_shape(y)[-1]//2, activation=None, name=f'{task}_dense')(y)
        y = tf.keras.layers.BatchNormalization(name=f'{task}_dense_batchnorm')(y)
        y = tf.keras.layers.Activation('relu', name=f'{task}_dense_relu')(y)        
        y = tf.keras.layers.Dropout(0.2)(y)
        if classes == 2:
            y = tf.keras.layers.Dense(1, activation='sigmoid', name=name)(y)
        else:
            y = tf.keras.layers.Dense(
                classes, activation='softmax', name=name)(y)
        model = tf.keras.Model(inputs=input_tensor, outputs=y)
        outputs.append(y)
        task_models[task] = model

    if really_new_tasks is not None and new_nb_classes is not None:
        for task, classes in zip(really_new_tasks, new_nb_classes):
            logger.info(f'Building model for {task} with {classes} classes...')
            adapter_rnn._add_new_task(task)
            y = adapter_rnn(x, task, mask)
            # y = apply_zero_mask([pooled_mask,
            #                      y])  # zero out silence activations
            y = tf.keras.layers.Dense(K.int_shape(y)[-1]//2, activation=None, name=f'{task}_dense')(y)
            y = tf.keras.layers.BatchNormalization(name=f'{task}_dense_batchnorm')(y)
            y = tf.keras.layers.Activation('relu', name=f'{task}_dense_relu')(y)    
            y = tf.keras.layers.Dropout(0.2)(y)
            if classes == 2:
                y = tf.keras.layers.Dense(
                    1, activation='sigmoid', name=task)(y)
            else:
                y = tf.keras.layers.Dense(
                    classes, activation='softmax', name=task)(y)
            model = tf.keras.Model(inputs=input_tensor, outputs=y)
            outputs.append(y)
            task_models[task] = model
    shared_model = tf.keras.Model(inputs=input_tensor, outputs=outputs)

    if initial_weights is not None:
        shared_model, task_models = load_and_assert_loaded(shared_model, task_models, initial_weights)
    return task_models, shared_model


def load_and_assert_loaded(shared_model, task_models, initial_weights):
    preloaded_layers = shared_model.layers.copy()
    shared_weights_pre_load, shared_names_pre_load = [], []
    for layer in preloaded_layers:
        if not layer.trainable:
            logger.debug(f'Appending weights of {layer.name} with shape before load.')
            shared_names_pre_load.append(layer.name)
            shared_weights_pre_load.append(layer.get_weights())
    logger.info('Loading weights from pre-trained model...')
    shared_model.load_weights(initial_weights, by_name=True)
    logger.info('Finished.')

    shared_weights_post_load = []
    shared_names_post_load = []
    for layer in shared_model.layers:
        if not layer.trainable:
            logger.debug(f'Appending weights of {layer.name} after load.')
            shared_names_post_load.append(layer.name)
            shared_weights_post_load.append(layer.get_weights())
    shared_weights_single_task = []
    shared_names_single_task = []
    single_task_model = task_models[list(
        task_models.keys())[0]]
    for layer in single_task_model.layers:
        if not layer.trainable:
            logger.debug(
                f'Appending weights of {layer.name} of single task model after load.')
            shared_names_single_task.append(layer.name)
            shared_weights_single_task.append(layer.get_weights())
    loaded, not_loaded, errors = 0, 0, 0
    assert shared_names_pre_load == shared_names_post_load and shared_names_post_load == shared_names_single_task, f'Layer name mistmatch: {shared_names_pre_load, shared_names_post_load, shared_names_single_task}.'
    for pre, pre_n, post, post_n, single, single_n in zip(shared_weights_pre_load, shared_names_pre_load, shared_weights_post_load, shared_names_post_load, shared_weights_single_task, shared_names_single_task):
        if array_list_equal(post, pre):
            not_loaded += 1
            logger.debug(
                f'Not loaded weights for layer {pre_n}: Total not loaded: {not_loaded}')
        elif array_list_equal(single, pre):
            not_loaded += 1
            logger.debug(
                f'Not loaded weights for layer {pre_n}: Total not loaded: {not_loaded}')
        elif array_list_equal(post, single):
            loaded += 1
            logger.debug(
                f'Loaded weights for layer {pre_n}. Total loaded: {loaded}')
        else:
            errors += 1
            logger.debug(
                f'Something went wrong with {pre_n, post_n, single_n}: {errors}')
    logger.info(
        f'Weights for {loaded} layers have been loaded from pre-trained model {initial_weights}.')
    return shared_model, task_models
