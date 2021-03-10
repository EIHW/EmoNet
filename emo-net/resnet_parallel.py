import sys
sys.path.append("..")
import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

import logging
import h5py
import tensorflow as tf
tf.get_logger().setLevel(logging.INFO)

from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Input, Add, Activation, Dropout, Flatten, dot, Dense, concatenate, ZeroPadding3D, ZeroPadding2D, Reshape, Lambda, Dropout, Layer, Permute, Masking, AveragePooling1D, RepeatVector
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, MaxPooling1D, AveragePooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dot, Multiply
from tensorflow.keras.layers import LSTM, GRU
from tensorflow.keras.layers import Bidirectional, TimeDistributed
from tensorflow.keras.optimizers import SGD
from sklearn.utils import class_weight
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2, L1L2
from tensorboard.plugins.hparams import api as hp
from .loader import AudioDataGenerator
from .metrics import ClassificationMetricCallback, KERAS_METRIC_QUANTITIES, KERAS_METRIC_MODES, UAR
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, TerminateOnNaN, EarlyStopping
from os.path import join
from os import makedirs
from .input_layers import LogMelgramLayer
from tqdm import tqdm
from .attention import SeqSelfAttention, SeqWeightedAttention
from tensorflow.keras.layers import LayerNormalization
from .losses import categorical_focal_loss, binary_focal_loss

import logging
logger = logging.getLogger(__name__)

kernel_regularizer = l2(1e-5)
rnn_regularizer = L1L2(1e-5)

channel_axis = 1 if K.image_data_format() == "channels_first" else -1


class BasicBlocksWithBetterNames(object):
    def __init__(self,
                 filters,
                 factor,
                 strides=2,
                 dropout1=0,
                 dropout2=0,
                 shortcut=True,
                 learnall=True,
                 tasks=['IEMOCAP-4cl'],
                 **kwargs):
        self.filters = filters
        self.factor = factor
        self.strides = strides
        self.dropout1 = Dropout(dropout1)
        self.dropout2 = Dropout(dropout2)
        self.shortcut = shortcut
        self.learnall = learnall
        self.tasks = tasks
        self.conv_task1 = ConvTasksWithBetterLayerNames(filters,
                                                        factor,
                                                        strides=strides,
                                                        learnall=learnall,
                                                        dropout=dropout1,
                                                        tasks=tasks,
                                                        **kwargs)
        self.conv_task2 = ConvTasksWithBetterLayerNames(filters,
                                                        factor,
                                                        strides=1,
                                                        learnall=learnall,
                                                        dropout=dropout2,
                                                        tasks=tasks,
                                                        **kwargs)

        self.relu = Activation('relu')
        if self.shortcut:
            self.avg_pool = AveragePooling2D((2, 2), padding='same')
            self.lmbda = Lambda(lambda x: x * 0)
        self.add = Add()

    def __call__(self, input_tensor, task):
        residual = input_tensor
        x = self.conv_task1(input_tensor, task=task)
        x = self.relu(x)
        x = self.conv_task2(x, task=task)
        if self.shortcut:
            residual = self.avg_pool(residual)
            residual0 = self.lmbda(residual)
            residual = concatenate([residual, residual0], axis=-1)
        x = self.add([residual, x])
        x = self.relu(x)
        return x

    def _add_new_task(self, task):
        self.conv_task1._add_new_task(task)
        self.conv_task2._add_new_task(task)


class BasicBlocks(object):
    def __init__(self,
                 filters,
                 factor,
                 strides=2,
                 dropout1=0,
                 dropout2=0,
                 shortcut=True,
                 learnall=True,
                 tasks=['IEMOCAP-4cl'],
                 **kwargs):
        self.filters = filters
        self.factor = factor
        self.strides = strides
        self.dropout1 = Dropout(dropout1)
        self.dropout2 = Dropout(dropout2)
        self.shortcut = shortcut
        self.learnall = learnall
        self.tasks = tasks
        self.conv_task1 = ConvTasks(filters,
                                    factor,
                                    strides=strides,
                                    learnall=learnall,
                                    dropout=dropout1,
                                    tasks=tasks)
        self.conv_task2 = ConvTasks(filters,
                                    factor,
                                    strides=1,
                                    learnall=learnall,
                                    dropout=dropout2,
                                    tasks=tasks)

        self.relu = Activation('relu')
        if self.shortcut:
            self.avg_pool = AveragePooling2D((2, 2), padding='same')
            self.lmbda = Lambda(lambda x: x * 0)
        self.add = Add()

    def __call__(self, input_tensor, task):
        residual = input_tensor
        x = self.conv_task1(input_tensor, task=task)
        x = self.relu(x)
        x = self.conv_task2(x, task=task)
        if self.shortcut:
            residual = self.avg_pool(residual)
            residual0 = self.lmbda(residual)
            residual = concatenate([residual, residual0], axis=-1)
        x = self.add([residual, x])
        x = self.relu(x)
        return x

    def _add_new_task(self, task):
        self.conv_task1._add_new_task(task)
        self.conv_task2._add_new_task(task)


class ConvTasksWithBetterLayerNames(object):
    def __init__(self,
                 filters,
                 factor=1,
                 strides=1,
                 learnall=True,
                 dropout=0,
                 tasks=['IEMOCAP-4cl', 'GEMEP'],
                 reuse_batchnorm=False,
                 **kwargs):
        self.filters = filters
        self.factor = factor
        self.strides = strides
        self.learnall = learnall
        self.dropout = Dropout(dropout)
        self.tasks = tasks
        self.reuse_batchnorm = reuse_batchnorm

        # shared parameters
        self.conv2d = Convolution2D(self.filters * self.factor, (3, 3),
                                    strides=self.strides,
                                    padding='same',
                                    kernel_initializer='he_normal',
                                    use_bias=False,
                                    trainable=self.learnall,
                                    kernel_regularizer=kernel_regularizer)

        # task specificparameters
        self.res_adapts = {}
        self.add = Add()
        self.bns = {}
        self.core_bn = BatchNormalization(
            axis=channel_axis,
            name=f'core_{self.conv2d.name}_batch_normalization')
        for task in self.tasks:
            self._add_new_task(task)

    def __call__(self, input_tensor, task):
        in_t = input_tensor
        if task is None:
            in_t = self.dropout(in_t) 
        x = self.conv2d(in_t)
        if task is not None:
            adapter_in = self.dropout(in_t)
            res_adapt = self.res_adapts[task](adapter_in)
            x = self.add([x, res_adapt])
        if self.reuse_batchnorm or task is None:
            x = self.core_bn(x)
        else:
            x = self.bns[task](x)
        return x

    def _add_new_task(self, task):
        assert task not in self.bns, 'Task already exists!'
        self.res_adapts[task] = Convolution2D(
            self.filters * self.factor, (1, 1),
            padding='valid',
            kernel_initializer='he_normal',
            strides=self.strides,
            use_bias=False,
            kernel_regularizer=kernel_regularizer,
            name=f'{task}_{self.conv2d.name}_adapter')
        
        if not self.reuse_batchnorm:
            self.bns[task] = BatchNormalization(
                axis=channel_axis,
                name=f'{task}_{self.conv2d.name}_batch_normalization')


class ConvTasks(object):
    def __init__(self,
                 filters,
                 factor=1,
                 strides=1,
                 learnall=True,
                 dropout=0,
                 tasks=['IEMOCAP-4cl', 'GEMEP'],
                 **kwargs):
        self.filters = filters
        self.factor = factor
        self.strides = strides
        self.learnall = learnall
        self.dropout = Dropout(dropout)
        self.tasks = tasks

        # shared parameters
        self.conv2d = Convolution2D(self.filters * self.factor, (3, 3),
                                    strides=self.strides,
                                    padding='same',
                                    kernel_initializer='he_normal',
                                    use_bias=False,
                                    trainable=self.learnall,
                                    kernel_regularizer=kernel_regularizer)

        # task specificparameters
        self.res_adapts = {}
        self.add = Add()
        self.bns = {}
        for task in self.tasks:
            self._add_new_task(task)

    def __call__(self, input_tensor, task):
        x = self.conv2d(input_tensor)
        adapter_in = self.dropout(input_tensor)
        res_adapt = self.res_adapts[task](adapter_in)
        x = self.add([x, res_adapt])
        x = self.bns[task](x)
        return x

    def _add_new_task(self, task):
        assert task not in self.bns, 'Task already exists!'
        self.res_adapts[task] = Convolution2D(
            self.filters * self.factor, (1, 1),
            padding='valid',
            kernel_initializer='he_normal',
            strides=self.strides,
            use_bias=False,
            kernel_regularizer=kernel_regularizer)
        self.bns[task] = BatchNormalization(axis=channel_axis)


class ResNetWithBetterLayerNames(object):
    def __init__(self,
                 filters=32,
                 factor=1,
                 N=2,
                 verbose=1,
                 learnall=True,
                 dropout1=0,
                 dropout2=0,
                 tasks=['IEMOCAP-4cl', 'GEMEP'],
                 reuse_batchnorm=False):
        self.filters = filters
        self.factor = factor
        self.N = N
        self.learnall = learnall
        self.dropout1 = dropout1
        self.dropout2 = dropout2
        self.tasks = tasks
        self.reuse_batchnorm = reuse_batchnorm

        self.input_core_bn = BatchNormalization(axis=channel_axis, name=f'core_input_batch_normalization')
        self.input_bns = {
            task: BatchNormalization(axis=channel_axis,
                                     name=f'{task}_input_batch_normalization')
            for task in self.tasks
        }
        
        # conv blocks
        self.pre_conv = ConvTasksWithBetterLayerNames(filters=self.filters,
                                                      factor=factor,
                                                      strides=1,
                                                      learnall=learnall,
                                                      tasks=self.tasks,
                                                      reuse_batchnorm=reuse_batchnorm)
        self.blocks = []
        self.nb_conv = 1
        for i in range(1, 4):
            block = BasicBlocksWithBetterNames(self.filters * (2**i),
                                               self.factor,
                                               strides=2,
                                               dropout1=self.dropout1,
                                               dropout2=self.dropout2,
                                               shortcut=True,
                                               learnall=self.learnall,
                                               tasks=self.tasks,
                                               reuse_batchnorm=reuse_batchnorm)
            self.blocks.append(block)
            for j in range(N - 1):
                block = BasicBlocksWithBetterNames(filters=self.filters *
                                                   (2**i),
                                                   factor=self.factor,
                                                   strides=1,
                                                   dropout1=self.dropout1,
                                                   dropout2=self.dropout2,
                                                   shortcut=False,
                                                   learnall=self.learnall,
                                                   tasks=self.tasks,
                                                   reuse_batchnorm=reuse_batchnorm)
                self.blocks.append(block)
                self.nb_conv += 2
        self.nb_conv += 6

        # bns and relus
        self.relu = Activation('relu')
        self.bns = {
            task: BatchNormalization(axis=channel_axis,
                                     name=f'{task}_final_batch_normalization')
            for task in self.tasks
        }
        self.core_bn = BatchNormalization(axis=channel_axis,
                                     name=f'core_final_batch_normalization')

    def _add_new_task(self, task):
        assert task not in self.bns, f'Task {task} already exists!'
        self.pre_conv._add_new_task(task)
        for block in self.blocks:
            block._add_new_task(task)
        if not self.reuse_batchnorm:
            self.bns[task] = BatchNormalization(axis=channel_axis, name=f'{task}_final_batch_normalization')
            self.input_bns[task] = BatchNormalization(axis=channel_axis, name=f'{task}_input_batch_normalization')

    def __call__(self, input_tensor, task):
        if task is None or self.reuse_batchnorm:
            x = self.input_core_bn(input_tensor)
        else:
            x = self.input_bns[task](input_tensor)
        x = self.pre_conv(x, task=task)
        for block in self.blocks:
            x = block(x, task=task)
        if task is None or self.reuse_batchnorm:
            x = self.core_bn(x)
        else:
            x = self.bns[task](x)
        x = self.relu(x)
        return x


class ResNet(object):
    def __init__(self,
                 filters=32,
                 factor=1,
                 N=2,
                 verbose=1,
                 learnall=True,
                 dropout1=0,
                 dropout2=0,
                 tasks=['IEMOCAP-4cl', 'GEMEP'],
                 **kwargs):
        self.filters = filters
        self.factor = factor
        self.N = N
        self.learnall = learnall
        self.dropout1 = dropout1
        self.dropout2 = dropout2
        self.tasks = tasks

        # conv blocks
        self.pre_conv = ConvTasks(filters=self.filters,
                                  factor=factor,
                                  strides=1,
                                  learnall=learnall,
                                  tasks=self.tasks)
        self.blocks = []
        self.nb_conv = 1
        for i in range(1, 4):
            block = BasicBlocks(self.filters * (2**i),
                                self.factor,
                                strides=2,
                                dropout1=self.dropout1,
                                dropout2=self.dropout2,
                                shortcut=True,
                                learnall=self.learnall,
                                tasks=self.tasks)
            self.blocks.append(block)
            for j in range(N - 1):
                block = BasicBlocks(filters=self.filters * (2**i),
                                    factor=self.factor,
                                    strides=1,
                                    dropout1=self.dropout1,
                                    dropout2=self.dropout2,
                                    shortcut=False,
                                    learnall=self.learnall,
                                    tasks=self.tasks)
                self.blocks.append(block)
                self.nb_conv += 2
        self.nb_conv += 6

        # bns and relus
        self.relu = Activation('relu')
        self.bns = {
            task: BatchNormalization(axis=channel_axis)
            for task in self.tasks
        }

    def _add_new_task(self, task):
        assert task not in self.bns, f'Task {task} already exists!'
        self.pre_conv._add_new_task(task)
        for block in self.blocks:
            block._add_new_task(task)
        self.bns[task] = BatchNormalization(axis=channel_axis)

    def __call__(self, input_tensor, task):
        x = self.pre_conv(input_tensor, task=task)
        for block in self.blocks:
            x = block(x, task=task)
        x = self.bns[task](x)
        x = self.relu(x)
        return x


def avgpool(x, n_classes, name):
    x = AveragePooling2D((8, 8))(x)
    x = Flatten()(x)
    return x



def attention_2d(x, n_classes, attention_size, name, mask=None):
    x = Reshape(target_shape=(-1, K.int_shape(x)[-1]))(x)
    #x = Masking(mask_value=0.)(x)
    x = SeqWeightedAttention()(x, mask=mask)
    return x


class RNNWithAdapters(object):
    def __init__(self,
                 input_dims,
                 hidden_size=512,
                 learnall=True,
                 dropout=0.2,
                 input_projection_factor=4,
                 adapter_projection_factor=2,
                 tasks=[],
                 recurrent_cell='lstm',
                 bidirectional=False,
                 **kwargs):

        self.hidden_size = hidden_size
        self.learnall = learnall
        self.dropout = dropout
        self.tasks = tasks
        self.input_dims = input_dims
        self.adapter_projection_factor = adapter_projection_factor
        self.input_projection_factor = input_projection_factor
        self.tasks = tasks
        self.recurrent_cell=recurrent_cell
        self.bidirectional = bidirectional

        self.reorder_dims = Permute((2, 1, 3))
        self.reshape = Reshape(target_shape=(-1,
                                             input_dims[1] * input_dims[3]))
        self.projection = Dense(input_dims[1] * input_dims[3] // self.input_projection_factor,
                                activation=None,
                                trainable=learnall,
                                kernel_regularizer=kernel_regularizer)
        #self.mask = Masking(mask_value=0.)
        self.rnn1 = GRU(self.hidden_size,
                          dropout=self.dropout,
                          return_sequences=True,
                          trainable=learnall, kernel_regularizer=rnn_regularizer) if self.recurrent_cell.lower() == 'gru' else LSTM(self.hidden_size,
                          dropout=self.dropout,
                          return_sequences=True,
                          trainable=learnall, kernel_regularizer=rnn_regularizer)
        if self.bidirectional:
            self.rnn1 = Bidirectional(self.rnn1)
            
        self.adapter_hidden_size = self.hidden_size *2 if self.bidirectional else self.hidden_size
        self.adapters = {
            task: RNNAdapter(self.adapter_hidden_size, self.adapter_projection_factor,
                             task)
            for task in tasks
        }
        self.add = Add()

        
        self.rnn2 = GRU(self.hidden_size,
                          dropout=self.dropout,
                          return_sequences=True,
                          trainable=learnall, kernel_regularizer=rnn_regularizer) if self.recurrent_cell.lower() == 'gru' else LSTM(self.hidden_size,
                          dropout=self.dropout,
                          return_sequences=True,
                          trainable=learnall, kernel_regularizer=rnn_regularizer)
        if self.bidirectional:
            self.rnn2 = Bidirectional(self.rnn2)
        
        self.weighted_attentions = {task: SeqWeightedAttention(trainable=True, name=f'{task}_seq_weighted_attention') for task in tasks}

    def __call__(self, x, task, mask=None):
        x = self.reorder_dims(x)
        x = self.reshape(x)
        #x = self.mask(x)
        x = self.projection(x)
        x = self.rnn1(x, mask=mask)
        adapter = self.adapters[task](x)
        x = self.add([x, adapter])
        #x = self.selfattention(x, mask=mask)
        x = self.rnn2(x, mask=mask)
        x = self.weighted_attentions[task](x, mask=mask)
        return x

    def _add_new_task(self, task):
        assert task not in self.adapters, f'Task {task} already exists!'
        self.adapters[task] = RNNAdapter(self.hidden_size,
                                         self.adapter_projection_factor, task)
        self.weighted_attentions[task] = SeqWeightedAttention(trainable=True, name=f'{task}_seq_weighted_attention')


class RNNAdapter(object):
    def __init__(self, input_size, downprojection=4, name='IEMOCAP'):
        self.input_size = input_size
        self.downprojection_factor = downprojection
        self.name = name
        self.layer_norm = TimeDistributed(LayerNormalization(),
                                          name=f'{name}_rnn_adapter_layer_norm')
        self.downprojection = TimeDistributed(Dense(
            self.input_size // self.downprojection_factor, activation='relu', use_bias=False, kernel_regularizer=kernel_regularizer),
                                              name=f'{name}_rnn_adapter_downprojection')
        self.upprojection = TimeDistributed(Dense(self.input_size, use_bias=False, kernel_regularizer=kernel_regularizer),
                                            name=f'{name}_rnn_adapter_upprojection')
        self.selfattention = SeqSelfAttention(
            attention_activation='sigmoid',
            kernel_regularizer=kernel_regularizer,
            use_attention_bias=False,
            trainable=True,
            name=f'{name}_rnn_adapter_attention')

    def __call__(self, x):
        x = self.layer_norm(x)
        x = self.downprojection(x)
        x = self.upprojection(x)
        x = self.selfattention(x)
        return x


def infer_tasks_from_weightfile(initial_weights):
    with h5py.File(initial_weights) as f:
        base_tasks = []
        base_nb_classes = []
        for k in f['model_weights']:
            prefices = ('activation', 'add', 'average_pooling',
                        'batch_normalization', 'concat', 'conv2d', 'dropout',
                        'dense', 'flatten', 'input', 'lambda',
                        'normalization2d', 'reshape', 'trainable_stft', 'apply_zero_mask', 'core', 'lstm', 'masking', 'permute', 'pooled', 'seq', 'zero_mask', 'adapter', 'mask', 'expand')
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
        return weight_dict['kernel:0'].shape[1]
    elif 'softmax' in weight_dict:
        return _find_n_classes(weight_dict['softmax'], task)
    elif task in weight_dict:
        return _find_n_classes(weight_dict[task], task)


class ComputeMask(tf.keras.layers.Layer):
    def __init__(self, num_fft, hop_length, **kwargs):
        super(ComputeMask, self).__init__(**kwargs)
        self.num_fft = num_fft
        self.hop_length = hop_length
    
    def call(self, x):
        frames = tf.signal.frame(x, self.num_fft, self.hop_length, pad_end=False,
                            axis=-1,
                            name=None)
        non_zeros = tf.math.count_nonzero(frames, axis=-1)
        mask = tf.not_equal(non_zeros, 0)
        return mask
    
    def get_config(self):
        config = {
            'num_fft': self.num_fft,
            'hop_length': self.hop_length,
        }
        base_config = super(ComputeMask, self).get_config()
        return dict(list(config.items()) + list(base_config.items()))


class PoolMask(tf.keras.layers.Layer):
    def __init__(self, pool_size, **kwargs):
        super(PoolMask, self).__init__(**kwargs)
        self.pool_size = pool_size
    
    def call(self, x):
        x = tf.expand_dims(x, -1)
        x = tf.cast(x, dtype='int8')
        x = tf.nn.pool(x, self.pool_size, pooling_type='MAX', padding='SAME', strides=self.pool_size)
        x = K.batch_flatten(x)
        x = tf.not_equal(x, 0)
        return x
    
    def get_config(self):
        config = {
            'pool_size': self.pool_size,
        }
        base_config = super(PoolMask, self).get_config()
        return dict(list(config.items()) + list(base_config.items()))

def create_mask(x):
    input_tensor = x
    x = tf.math.count_nonzero(x, 1, keepdims=True)
    x = K.clip(x, min_value=0, max_value=1)
    x = K.cast(x, dtype='float32')
    x = K.repeat_elements(x, input_tensor.shape[1], 1)
    return x

def pool_mask(mask, pool_size=(8,)):
    x = tf.expand_dims(mask, -1)
    x = tf.cast(x, dtype='int8')
    x = tf.nn.pool(x, pool_size, pooling_type='MAX', padding='SAME', strides=pool_size)
    x = K.batch_flatten(x)
    x = tf.not_equal(x, 0)
    return x

def compute_mask(inputs, num_fft, hop_length):
    frames = tf.signal.frame(inputs, num_fft, hop_length, pad_end=False,
                            axis=-1,
                            name=None)
    non_zeros = tf.math.count_nonzero(frames, axis=-1)
    mask = tf.not_equal(non_zeros, 0)
    return mask

def create_multi_task_resnets(input_dim,
                              filters=32,
                              factor=1,
                              N=4,
                              verbose=1,
                              learnall=True,
                              learnall_classifier=True,
                              dropout1=0,
                              dropout2=0,
                              rnn_dropout=0.2,
                              base_tasks=['EMO-DB', 'GEMEP'],
                              new_tasks=None,
                              base_nb_classes=[6, 10],
                              new_nb_classes=None,
                              initial_weights=None,
                              random_noise=0.1,
                              feature_extractor='cnn',
                              input_features='melspecs',
                              classifier='FCNAttention',
                              variable_duration=False,
                              reuse_batchnorm=False):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    if base_tasks is None:
        assert initial_weights is not None, f'Either base tasks or initial weights have to be specified!'
        if verbose:
            logger.info(
                'Trying to determine trained tasks from initial weights...')
        base_tasks, base_nb_classes = infer_tasks_from_weightfile(
            initial_weights)

    input_tensor = Input(shape=input_dim)
    if variable_duration:
        input_reshaped = Reshape(target_shape=(-1, ))(input_tensor)
    else:
        input_reshaped = Reshape(target_shape=(input_dim[0], ))(input_tensor)

    input_features = LogMelgramLayer(num_fft=1024,
                        hop_length=256,
                        sample_rate=16000,
                        f_min=20,
                        f_max=8000,
                        num_mels=128,
                        eps=1e-6,
                        return_decibel=False,
                        name='trainable_stft')
    x = input_features(input_reshaped)
    mask = ComputeMask(input_features.num_fft, input_features.hop_length)(input_reshaped)
    pooled_mask = PoolMask((8,))(mask)
    # def compute_and_pool_mask(x):
    #     print(x)
    #     mask = compute_mask(x, input_features.num_fft, input_features.hop_length)
    #     return pool_mask(mask, pool_size=(8,))
    # pooled_mask = compute_and_pool_mask(input_reshaped)

    expand_dims = Lambda(lambda x: tf.expand_dims(x, 3), name='expand_input_dims')
    x = expand_dims(x)
    x = Permute((2, 1, 3))(x)
    #mask = input_features.compute_mask(input_reshaped)
    #mask = Lambda(create_mask, name='zero_mask')(x)
    #mask = K.print_tensor(mask, 'mask')
    # pooled_mask = AveragePooling2D((8, 8),
    #                                name='pooled_zero_mask',
    #                                padding='same')(mask)
    #apply_zero_mask = Multiply(name='apply_zero_mask')
    #x = Normalization2D(str_axis='freq')(x)
    adapter_resnet = ResNetWithBetterLayerNames(filters=filters,
                                                factor=factor,
                                                N=N,
                                                verbose=verbose,
                                                learnall=learnall,
                                                dropout1=dropout1,
                                                dropout2=dropout2,
                                                #tasks=base_tasks,
                                                tasks=[] if len(base_tasks) == 1 else base_tasks)

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
    adapter_rnn = None
    for task, classes in zip(base_tasks, base_nb_classes):
        logger.info(f'Building model for {task} with {classes} classes...')
        #y = adapter_resnet(x, task)
        adapters_in_resnet = task if len(base_tasks) > 1 else None
        y = adapter_resnet(x, adapters_in_resnet)
        #y = apply_zero_mask([pooled_mask, y])  # zero out silence activations
        if initial_weights is not None:  # might need new last dense layer
            name = f'{task}_1'
        else:
            name = task
        if classifier == 'avgpool':
            y = avgpool(y, n_classes=classes, name=name)
        if classifier == 'FCNAttention':
            y = FCNAttention(classes, K.int_shape(y)[-1], 0.3, name)(y)
        if classifier == 'attention':
            y = attention_classifier(y, classes, K.int_shape(y)[-1], name)
        if classifier == 'attention2d':
            y = attention_2d(y, classes, K.int_shape(y)[-1], name, mask=pooled_mask)
        if classifier == 'rnn':
            if adapter_rnn is None:
                adapter_rnn = RNNWithAdapters(K.int_shape(y),
                                              hidden_size=K.int_shape(y)[-1],
                                              learnall=learnall_classifier,
                                              dropout=rnn_dropout,
                                              input_projection_factor=4,
                                              adapter_projection_factor=4,
                                              #tasks=base_tasks,
                                              tasks=base_tasks)
            #y = adapter_rnn(y, task)
            y = adapter_rnn(y, task, mask=pooled_mask)

        if classes == 2:
            y = Dense(1, activation='sigmoid', name=name)(y)
        else:
            y = Dense(classes, activation='softmax', name=name)(y)
        model = Model(inputs=input_tensor, outputs=y)
        outputs.append(y)
        task_models[task] = model

    if really_new_tasks is not None and new_nb_classes is not None:
        for task, classes in zip(really_new_tasks, new_nb_classes):
            logger.info(f'Building model for {task} with {classes} classes...')
            adapter_resnet._add_new_task(task)
            y = adapter_resnet(x, task)
            # y = apply_zero_mask([pooled_mask,
            #                      y])  # zero out silence activations

            if classifier == 'avgpool':
                y = avgpool(y, n_classes=classes, name=task)
            if classifier == 'FCNAttention':
                y = FCNAttention(classes, K.int_shape(y)[-1], 0.3, task)(y)
            if classifier == 'attention':
                y = attention_classifier(y, classes, K.int_shape(y)[-1], task)
            if classifier == 'attention2d':
                y = attention_2d(y, classes, K.int_shape(y)[-1], task)
            if classifier == 'rnn':
                adapter_rnn._add_new_task(task)
                y = adapter_rnn(y, task)
            if classes == 2:
                y = Dense(1, activation='sigmoid', name=task)(y)
            else:
                y = Dense(classes, activation='softmax', name=task)(y)
            model = Model(inputs=input_tensor, outputs=y)
            outputs.append(y)
            task_models[task] = model
    shared_model = Model(inputs=input_tensor, outputs=outputs)
    #print(shared_model.get_config())

    
    if initial_weights is not None:
        preloaded_layers = shared_model.layers.copy()
        shared_weights_pre_load, shared_names_pre_load = [], []
        for layer in preloaded_layers:
            if not layer.trainable:
                logger.debug(f'Appending weights of {layer.name} before load.')
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
        single_task_model = task_models[new_tasks[0]] if new_tasks else task_models[list(task_models.keys())[0]]
        for layer in single_task_model.layers:
            if not layer.trainable:
                logger.debug(f'Appending weights of {layer.name} of single task model after load.')
                shared_names_single_task.append(layer.name)
                shared_weights_single_task.append(layer.get_weights())
        loaded, not_loaded, errors = 0, 0, 0
        assert shared_names_pre_load == shared_names_post_load and shared_names_post_load == shared_names_single_task, f'Layer name mistmatch: {shared_names_pre_load, shared_names_post_load, shared_names_single_task}.'
        for pre, pre_n, post, post_n, single, single_n in zip (shared_weights_pre_load, shared_names_pre_load, shared_weights_post_load, shared_names_post_load, shared_weights_single_task, shared_names_single_task):
            if array_list_equal(post, pre):
                not_loaded += 1
                logger.debug(f'Not loaded weights for layer {pre_n}: Total not loaded: {not_loaded}')
            elif array_list_equal(single, pre):
                not_loaded += 1
                logger.debug(f'Not loaded weights for layer {pre_n}: Total not loaded: {not_loaded}')
            elif array_list_equal(post, single):
                loaded += 1
                logger.debug(f'Loaded weights for layer {pre_n}. Total loaded: {loaded}')
            else:
                errors += 1
                logger.debug(f'Something went wrong with {pre_n, post_n, single_n}: {errors}')
        logger.info(f'Weights for {loaded} layers have been loaded from pre-trained model {initial_weights}.')
    return task_models, shared_model

def array_list_equal(a_list, b_list):
    if type(a_list) == list and type(b_list) == list:
        if len(a_list) != len(b_list):
            return False
        else:
            for a, b in zip(a_list, b_list):
                if not np.array_equal(a,b):
                    return False
            return True
    elif type(a_list) == np.array and type(b_list) == np.array:
        return np.array_equal(a_list, b_list)
    else:
        return False

def determine_decay(generator, batch_size):
    if 10000 > len(generator) * batch_size >= 1000:
        decay = 0.0005
    elif 1000 > len(generator) * batch_size >= 500:
        decay = 0.002
    elif len(generator) * batch_size < 500:
        decay = 0.005
    else:
        decay = 1e-6
    return decay


def named_logs(model, logs):
    result = {}
    for l in zip(model.metrics_names, logs):
        result[l[0]] = l[1]
    return result


def train_single_task(
    initial_weights='/mnt/student/MauriceGerczuk/EmoSet/experiments/residual-adapters-emonet-revised/parallel/128mels/2s/scratch/N-2_factor-1-balancedClassWeights-True/GEMEP/weights_GEMEP.h5',
    batch_size=64,
    epochs=50,
    balanced_weights=True,
    window=6,
    N=2,
    factor=1,
    dropout1=0,
    dropout2=0.5,
    rnn_dropout=0.2,
    learnall_classifier=True,
    feature_extractor='cnn',
    input_features='melspecs',
    task='IEMOCAP-4cl',
    classifier='FCNAttention',
    loss='categorical_cross_entropy',
    directory='/mnt/nas/data_work/shahin/EmoSet/wavs-reordered/',
    train_csv='train.csv',
    val_csv='val.csv',
    test_csv='test.csv',
    experiment_base_path='/mnt/student/MauriceGerczuk/EmoSet/experiments/residual-adapters-emonet-revised',
    random_noise=None,
    learnall=False,
    last_layer_only=False,
    initial_learning_rate=0.1,
    optimizer=SGD,
    n_workers=5,
    patience=20,
    mode='adapters'):
    # hparams = {
    #     'batch_size': batch_size,
    #     'balanced_weights': balanced_weights,
    #     'window': window if window is not None else -1,
    #     'N': N,
    #     'factor': factor,
    #     'dropout1': dropout1,
    #     'dropout2': dropout2,
    #     'rnn_dropout': rnn_dropout,
    #     'learnall_classifier': learnall_classifier,
    #     'classifier': classifier,
    #     'loss': loss,
    #     'learnall': learnall,
    #     'last_layer_only': last_layer_only,
    #     'initial_learning_rate': initial_learning_rate,
    #     'optimizer': optimizer.__name__,
    #     'patience': patience
    # }
    variable_duration = False if classifier == 'avgpool' else True
    #variable_duration = False
    base_tasks = None
    base_nb_classes = None
    experiment_base_path = f"{join(experiment_base_path, f'Window-{window}s', mode, classifier)}"
    experiment_base_path = join(
        experiment_base_path,
        f'N-{N}_factor-{factor}-balancedClassWeights-{balanced_weights}-loss-{loss}-optimizer-{optimizer.__name__}-lr-{initial_learning_rate}-bs-{batch_size}-patience-{patience}-do1-{dropout1}-do2-{dropout2}-{"rd-"+str(rnn_dropout) if classifier == "rnn" else ""}-random_noise-{random_noise}'
    )
    train_generator = AudioDataGenerator(train_csv,
                                         directory,
                                         batch_size=batch_size,
                                         window=window,
                                         shuffle=True,
                                         sr=16000,
                                         time_stretch=None,
                                         pitch_shift=None,
                                         save_dir=None,
                                         val_split=None,
                                         subset='train',
                                         variable_duration=variable_duration)
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
    test_generator = AudioDataGenerator(test_csv,
                                        directory,
                                        batch_size=batch_size,
                                        window=window,
                                        shuffle=False,
                                        sr=16000,
                                        time_stretch=None,
                                        pitch_shift=None,
                                        save_dir=None,
                                        variable_duration=variable_duration)

    if initial_weights is not None and mode == 'adapters':
        new_tasks = [task]
        new_nb_classes = [len(set(train_generator.classes))]
    else:
        base_tasks = [task]
        base_nb_classes = [len(set(train_generator.classes))]
        new_tasks = []
        new_nb_classes = []

    if balanced_weights:
        class_weights = class_weight.compute_class_weight(
            'balanced', np.unique(train_generator.classes),
            train_generator.classes)
        class_weight_dict = dict(enumerate(class_weights))
        logger.info(f'Class weights: {class_weight_dict}')
    else:
        class_weight_dict = None
        logger.info('Not using class weights.')

    task_base_path = join(experiment_base_path, task)
    weights = join(task_base_path, "weights_" + task + ".h5")

    decay = determine_decay(train_generator, batch_size)

    x, _ = train_generator[0]
    if not variable_duration:
        init = x.shape[1:]
    else:
        init = (None, )
    models, shared_model = create_multi_task_resnets(
        init,
        filters=32,
        factor=factor,
        feature_extractor=feature_extractor,
        input_features=input_features,
        base_nb_classes=base_nb_classes,
        N=N,
        verbose=1,
        learnall=learnall,
        learnall_classifier=learnall_classifier,
        dropout1=dropout1,
        dropout2=dropout2,
        rnn_dropout=rnn_dropout,
        base_tasks=base_tasks,
        new_tasks=new_tasks,
        new_nb_classes=new_nb_classes,
        initial_weights=initial_weights,
        classifier=classifier,
        random_noise=random_noise,
        variable_duration=variable_duration,
        reuse_batchnorm=True)
    model = models[task]
    #model.load_weights(initial_weights, by_name=True)
    if last_layer_only:
        for layer in model.layers[:-1]:
            layer.trainable = False
    model.summary()
    #model.load_weights(initial_weights, by_name=True)

    tbCallBack = TensorBoard(log_dir=join(task_base_path, 'log'),
                             histogram_freq=0,
                             write_graph=True)
    #hpCallback = hp.KerasCallback(join(task_base_path, 'log', 'hparam_tuning'), hparams)
    mc = ModelCheckpoint(weights,
                         monitor='val_recall/macro/validation',
                         verbose=1,
                         save_best_only=True,
                         save_weights_only=False,
                         mode='max',
                         period=1)
    metric_callback = ClassificationMetricCallback(
        validation_generator=val_generator, dataset_name=task)
    metric_callback_test = ClassificationMetricCallback(
        validation_generator=test_generator, dataset_name=task, partition='test')

    lrs = [
        initial_learning_rate, initial_learning_rate * 0.1,
        initial_learning_rate * 0.01
    ]
    makedirs(task_base_path, exist_ok=True)
    stopped_epoch = 0
    best = 0
    patience = patience
    load = False
    if len(set(train_generator.classes)) == 2:
        loss = binary_focal_loss(
        ) if loss == "focal" else "binary_crossentropy"
    else:
        loss = categorical_focal_loss(
        ) if loss == "focal" else "categorical_crossentropy"
    for i, lr in enumerate(lrs):
        if optimizer.__name__ == SGD.__name__:
            opt = optimizer(lr=lr, decay=decay, momentum=0.9, nesterov=False)
        else:
            opt = optimizer(lr=lr)
        #opt = SGD(lr=lr, decay=decay, momentum=0.9, nesterov=False)
        model.compile(loss=loss, optimizer=opt, metrics=["acc"], experimental_run_tf_function=False)

        if load:
            model.load_weights(weights)
            logger.info("Model loaded.")
        early_stopper = EarlyStopping(monitor='val_recall/macro/validation',
                                      min_delta=0.005,
                                      patience=patience,
                                      verbose=1,
                                      mode='max',
                                      restore_best_weights=False,
                                      baseline=best)
        model.fit_generator(train_generator,
                            validation_data=val_generator,
                            epochs=epochs,
                            workers=n_workers // 2,
                            initial_epoch=stopped_epoch,
                            class_weight=class_weight_dict,
                            use_multiprocessing=True,
                            max_queue_size=n_workers * 2,
                            verbose=2,
                            callbacks=[
                                metric_callback, metric_callback_test,
                                early_stopper, tbCallBack, mc
                            ])
        load = True
        stopped_epoch = early_stopper.stopped_epoch
        best = early_stopper.best


def train_multi_task(
    batch_size=64,
    epochs=50,
    balanced_weights=True,
    window=2,
    N=2,
    factor=1,
    dropout1=0.5,
    dropout2=0.5,
    rnn_dropout=0.2,
    initial_learning_rate=0.1,
    tasks=[
        "AirplaneBehaviourCorpus", "AngerDetection", "BurmeseEmotionalSpeech",
        "CASIA", "ChineseVocalEmotions", "DanishEmotionalSpeech", "DEMoS",
        "EA-ACT", "EA-BMW", "EA-WSJ", "EMO-DB", "EmoFilm", "EmotiW-2014",
        "ENTERFACE", "EU-EmoSS", "FAU_AIBO", "GEMEP", "GVESS",
        "MandarinEmotionalSpeech", "MELD", "PPMK-EMO", "SIMIS", "SMARTKOM",
        "SUSAS", "TurkishEmoBUEE"
    ],
    loss='categorical_crossentropy',
    classifier='avgpool',
    directory='/mnt/nas/data_work/shahin/EmoSet/wavs-reordered/',
    experiment_base_path='/mnt/student/MauriceGerczuk/EmoSet/experiments/residual-adapters-emonet-revised',
    multi_task_setup='/mnt/student/MauriceGerczuk/EmoSet/multiTaskSetup-wavs-with-test/',
    steps_per_epoch=20,
    optimizer=SGD,
    random_noise=None):
    variable_duration = False if classifier == 'avgpool' else True
    experiment_base_path = f"{join(experiment_base_path, f'Window-{window}s', 'multi-task', '-'.join(tasks), classifier)}"
    experiment_base_path = join(
        experiment_base_path,
        f'N-{N}_factor-{factor}-balancedClassWeights-{balanced_weights}-loss-{loss}-optimizer-{optimizer.__name__}-lr-{initial_learning_rate}-bs-{batch_size}-spe-{steps_per_epoch}-epochs-{epochs}-do1-{dropout1}-do2-{dropout2}-{"rd-"+str(rnn_dropout) if classifier == "rnn" else ""}-random_noise-{random_noise}'
    )
    train_generators = [
        AudioDataGenerator(f'{multi_task_setup}/{task}/train.csv',
                           directory,
                           batch_size=batch_size,
                           window=window,
                           shuffle=True,
                           sr=16000,
                           time_stretch=None,
                           pitch_shift=None,
                           variable_duration=variable_duration,
                           save_dir=None,
                           val_split=None,
                           subset='train') for task in tasks
    ]
    val_generators = [
        AudioDataGenerator(f'{multi_task_setup}/{task}/val.csv',
                           directory,
                           batch_size=batch_size,
                           window=window,
                           shuffle=False,
                           sr=16000,
                           time_stretch=None,
                           variable_duration=variable_duration,
                           pitch_shift=None,
                           save_dir=None) for task in tasks
    ]
    test_generators = [
        AudioDataGenerator(f'{multi_task_setup}/{task}/test.csv',
                           directory,
                           batch_size=batch_size,
                           window=window,
                           shuffle=False,
                           sr=16000,
                           time_stretch=None,
                           variable_duration=variable_duration,
                           pitch_shift=None,
                           save_dir=None) for task in tasks
    ]

    if balanced_weights:
        class_weights = [
            class_weight.compute_class_weight('balanced', np.unique(t.classes),
                                              t.classes)
            for t in train_generators
        ]
        class_weight_dicts = [dict(enumerate(cw)) for cw in class_weights]
        logger.info(f'Class weights: {class_weight_dicts}')
    else:
        class_weight_dicts = [None] * len(tasks)
        logger.info('Not using class weights.')

    task_base_paths = [join(experiment_base_path, task) for task in tasks]
    weight_paths = [
        join(task_base_path, "weights_" + task + ".h5")
        for task_base_path, task in zip(task_base_paths, tasks)
    ]

    tbCallBacks = [
        TensorBoard(log_dir=join(task_base_path, 'log'),
                    histogram_freq=0,
                    write_graph=True) for task_base_path in task_base_paths
    ]

    metric_callbacks = [
        ClassificationMetricCallback(validation_generator=val_generator,
                                     period=10, dataset_name=task)
        for val_generator, task in zip(val_generators, tasks)
    ]
    metric_callbacks_test = [
        ClassificationMetricCallback(validation_generator=test_generator,
                                     partition='test',
                                     period=10,
                                     dataset_name=task)
        for test_generator, task in zip(test_generators, tasks)
    ]
    decays = [determine_decay(tg, batch_size) for tg in train_generators]

    #steps_per_epoch = 10
    x, _ = train_generators[0][0]
    if not variable_duration:
        init = x.shape[1:]
    else:
        init = (None, )

    lrs = [
        initial_learning_rate, initial_learning_rate * 0.1,
        initial_learning_rate * 0.01
    ]
    nb_classes = [len(tg.class_indices) for tg in train_generators]
    models, shared_model = create_multi_task_resnets(
        init,
        filters=32,
        factor=factor,
        base_nb_classes=nb_classes,
        N=N,
        verbose=1,
        learnall=True,
        dropout1=dropout1,
        dropout2=dropout2,
        rnn_dropout=rnn_dropout,
        learnall_classifier=True,
        base_tasks=tasks,
        classifier=classifier,
        variable_duration=variable_duration)
    shared_model.summary()
    for i, t in enumerate(tasks):
        tbCallBacks[i].set_model(models[t])
        metric_callbacks[i].set_model(models[t])
        metric_callbacks_test[i].set_model(models[t])

    for step, lr in enumerate(lrs):
        for i in tqdm(range(epochs * steps_per_epoch)):
            for t, task in enumerate(tasks):
                model = models[task]
                if i == 0:  # reset learning rate
                    if len(set(train_generators[t].classes)) == 2:
                        loss = binary_focal_loss(
                        ) if loss == "focal" else "binary_crossentropy"
                    else:
                        loss = categorical_focal_loss(
                        ) if loss == "focal" else "categorical_crossentropy"
                    if optimizer.__name__ == SGD.__name__:
                        opt = optimizer(lr=lr,
                                        decay=decays[t],
                                        momentum=0.9,
                                        nesterov=False)
                    else:
                        opt = optimizer(lr=lr)
                    model.compile(loss=loss, optimizer=opt, metrics=["acc"])
                if i % len(train_generators[t]) == 0:
                    train_generators[t].on_epoch_end()
                logs = model.train_on_batch(*train_generators[t][i])
                named_l = named_logs(model, logs)
                loss = named_l["loss"]
                logger.info(f'Step {i}: loss {loss} ({task})')
                if i % steps_per_epoch == 0:
                    metric_callbacks[t].on_epoch_end(
                        i + step * epochs * steps_per_epoch, named_l)
                    metric_callbacks_test[t].on_epoch_end(
                        i + step * epochs * steps_per_epoch, named_l)
                    model.save(weight_paths[t])
                tbCallBacks[t].on_epoch_end(
                    i + step * epochs * steps_per_epoch, named_l)
                shared_model.save(join(experiment_base_path, 'shared_model.h5'))
